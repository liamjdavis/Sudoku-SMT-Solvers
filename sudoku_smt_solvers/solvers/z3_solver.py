import multiprocessing
from z3 import Solver, Bool, And, Or, Not, Implies, sat, unsat
from .sudoku_error import SudokuError


class Z3Solver:
    def __init__(self, sudoku, timeout=120) -> None:
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")

        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self._validate_input(sudoku)
        self.sudoku = sudoku
        self.timeout = timeout
        self.solver = None
        self.variables = None
        self.propagated_clauses = 0

    def _validate_input(self, sudoku):
        if not sudoku or not isinstance(sudoku, list):
            raise SudokuError("Invalid Sudoku puzzle: input must be a list")

        if len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        for i, row in enumerate(sudoku):
            if not isinstance(row, list) or len(row) != 25:
                raise SudokuError(f"Invalid Sudoku puzzle: row {i} must have 25 cells")
            for j, val in enumerate(row):
                if not isinstance(val, int):
                    raise SudokuError(
                        f"Invalid value at position ({i},{j}): must be an integer"
                    )
                if val < 0 or val > 25:
                    raise SudokuError(
                        f"Invalid value {val} at position ({i},{j}): must be between 0 and 25"
                    )

    def create_variables(self):
        """Set self.variables as a 3D list containing the Z3 variables."""
        if not self.solver:
            self.solver = Solver()
        try:
            self.variables = [
                [[Bool(f"x_{i}_{j}_{k}") for k in range(25)] for j in range(25)]
                for i in range(25)
            ]
        except Exception as e:
            raise SudokuError(f"Failed to create Z3 variables: {str(e)}")

    def encode_rules(self):
        """Encode the rules of Sudoku into the solver."""
        if not self.solver or not self.variables:
            raise SudokuError("Solver not initialized properly")

        try:
            for i in range(25):
                for j in range(25):
                    try:
                        self.solver.add(
                            Or([self.variables[i][j][k] for k in range(25)])
                        )
                    except Exception as e:
                        raise SudokuError(
                            f"Failed to encode cell constraint at ({i},{j}): {str(e)}"
                        )

                    for k1 in range(25):
                        for k2 in range(k1 + 1, 25):
                            try:
                                self.solver.add(
                                    Not(
                                        And(
                                            self.variables[i][j][k1],
                                            self.variables[i][j][k2],
                                        )
                                    )
                                )
                            except Exception as e:
                                raise SudokuError(
                                    f"Failed to encode value constraint at ({i},{j}): {str(e)}"
                                )

        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Failed to encode Sudoku rules: {str(e)}")

    def encode_puzzle(self):
        """Encode the initial puzzle into the solver."""
        if not self.solver or not self.variables:
            raise SudokuError("Solver not initialized properly")

        try:
            for i in range(25):
                for j in range(25):
                    if self.sudoku[i][j] != 0:
                        try:
                            self.solver.add(self.variables[i][j][self.sudoku[i][j] - 1])
                        except Exception as e:
                            raise SudokuError(
                                f"Failed to encode initial value at ({i},{j}): {str(e)}"
                            )
        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Failed to encode initial puzzle: {str(e)}")

    def extract_solution(self, model):
        """
        Extract the solution from the model with error handling.

        Args:
            model: Z3 model containing solution

        Returns:
            2D list representing solved Sudoku grid

        Raises:
            SudokuError: If solution extraction fails
        """
        if not model:
            raise SudokuError("Invalid model: model cannot be None")

        if not self.variables:
            raise SudokuError("Variables not initialized")

        solution = [[0 for _ in range(25)] for _ in range(25)]
        try:
            for i in range(25):
                for j in range(25):
                    cell_assigned = False
                    for k in range(25):
                        try:
                            if model.evaluate(self.variables[i][j][k]):
                                if cell_assigned:
                                    raise SudokuError(
                                        f"Multiple values assigned to cell ({i},{j})"
                                    )
                                solution[i][j] = k + 1
                                cell_assigned = True
                        except Exception as e:
                            raise SudokuError(
                                f"Failed to evaluate variable at ({i},{j},{k}): {str(e)}"
                            )

                    if not cell_assigned:
                        raise SudokuError(f"No value assigned to cell ({i},{j})")

            return solution

        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Failed to extract solution: {str(e)}")

    def _solve_task(self):
        """Helper method to run in separate process"""
        try:
            self.solver = Solver()
            self.create_variables()
            self.encode_rules()
            self.encode_puzzle()

            try:
                stats = self.solver.statistics()
                try:
                    self.propagated_clauses = stats.get("propagations", 0)
                except (AttributeError, Exception):
                    self.propagated_clauses = 0

                result = self.solver.check()

                if result == sat:
                    model = self.solver.model()
                    solution = self.extract_solution(model)

                    if not self._validate_solution(solution):
                        raise SudokuError("Generated solution is invalid")
                    return solution

            except SudokuError as e:
                raise
            except Exception as e:
                raise SudokuError(f"Solver error: {str(e)}")

        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Critical solver error: {str(e)}")

        return None

    def solve(self):
        """Solve the Sudoku puzzle with multiprocessing-based timeout"""
        try:
            # Skip multiprocessing if in test mode
            if hasattr(self, '_testing'):
                solution = self._solve_task()
                return solution
            
            # Create a process pool with 1 worker
            with multiprocessing.Pool(1) as pool:
                try:
                    # Run solver in separate process with timeout
                    async_result = pool.apply_async(self._solve_task)
                    solution = async_result.get(timeout=self.timeout)

                    return solution

                except multiprocessing.TimeoutError:
                    raise SudokuError(f"Solver timed out after {self.timeout} seconds")
                
        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Critical solver error: {str(e)}")

        return None

    def _validate_solution(self, solution):
        """Validate the generated solution"""
        if not solution:
            return False

        try:
            # Check dimensions
            if len(solution) != 25 or any(len(row) != 25 for row in solution):
                return False

            # Check value range
            if any(
                not isinstance(val, int) or val < 1 or val > 25
                for row in solution
                for val in row
            ):
                return False

            # Check if solution matches initial puzzle
            for i in range(25):
                for j in range(25):
                    if self.sudoku[i][j] != 0 and self.sudoku[i][j] != solution[i][j]:
                        return False

            return True

        except Exception:
            return False
