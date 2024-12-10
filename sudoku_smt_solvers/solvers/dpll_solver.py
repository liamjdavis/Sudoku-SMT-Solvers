import multiprocessing

from .sudoku_error import SudokuError
from .utils import generate_cnf, get_var


class DPLLSolver:
    def __init__(self, sudoku, timeout=120) -> None:
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")

        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = 25
        self.propagated_clauses = 0
        self.timeout = timeout
        self.clauses = []
        self.assignments = {}

        # Generate CNF Form of Sudoku
        try:
            generate_cnf(self.size)
        except Exception as e:
            raise SudokuError(f"Error generating CNF: {e}")

    def find_unit_clause(self):
        """Find a unit clause in the current formula"""
        for clause in self.clauses:
            unassigned = []
            for lit in clause:
                if abs(lit) not in self.assignments:
                    unassigned.append(lit)
                elif (lit > 0 and self.assignments[abs(lit)]) or (
                    lit < 0 and not self.assignments[abs(lit)]
                ):
                    # Clause is satisfied
                    break
            else:
                if len(unassigned) == 1:
                    return unassigned[0]
        return None

    def find_pure_literal(self):
        """Find a pure literal in the current formula"""
        # Track both positive and negative occurrences of each variable
        occurrences = {}

        for clause in self.clauses:
            if any(var in self.assignments.keys() for var in clause):
                continue

            for lit in clause:
                var = abs(lit)
                if var not in self.assignments:  # Ignore already assigned variables
                    if var not in occurrences:
                        occurrences[var] = {"pos": False, "neg": False}
                    if lit > 0:
                        occurrences[var]["pos"] = True
                    else:
                        occurrences[var]["neg"] = True

        # Find a variable that appears only positively or only negatively
        for var, counts in occurrences.items():
            if counts["pos"] and not counts["neg"]:  # Purely positive literal
                return var
            if counts["neg"] and not counts["pos"]:  # Purely negative literal
                return -var

        return None

    def update_clauses(self, assignment):
        """Update clauses based on new assignment"""
        var = abs(assignment)
        value = assignment > 0
        self.assignments[var] = value

        new_clauses = []

        for clause in self.clauses:
            if any(
                lit > 0
                and self.assignments.get(abs(lit), False)
                or lit < 0
                and not self.assignments.get(abs(lit), True)
                for lit in clause
            ):
                self.propagated_clauses += 1

                continue

            new_clause = [lit for lit in clause if abs(lit) not in self.assignments]

            if new_clause:
                new_clauses.append(new_clause)

            elif not new_clause:
                self.propagated_clauses += 1
                return False

        self.clauses = new_clauses

        return True

    def dpll(self):
        """Main DPLL algorithm"""
        # Unit propagation
        while unit := self.find_unit_clause():
            if not self.update_clauses(unit):
                return False

        # Pure literal elimination
        while pure := self.find_pure_literal():
            if not self.update_clauses(pure):
                return False

        # All clauses satisfied
        if not self.clauses:
            return True

        # Choose variable
        var = abs(self.clauses[0][0])

        # Try True
        assignments_backup = self.assignments.copy()
        clauses_backup = [clause[:] for clause in self.clauses]
        if self.update_clauses(var) and self.dpll():
            return True

        # Try False
        self.assignments = assignments_backup
        self.clauses = clauses_backup
        if self.update_clauses(-var) and self.dpll():
            return True

        return False

    def _solve_task(self):
        """Helper method to run in separate process"""
        try:
            for i in range(self.size):
                for j in range(self.size):
                    if self.sudoku[i][j] != 0:
                        if not self.update_clauses(
                            get_var(i, j, self.sudoku[i][j], self.size)
                        ):
                            return None

            if self.dpll():
                solution = [[0] * self.size for _ in range(self.size)]
                for var, value in self.assignments.items():
                    if value:
                        row = (var - 1) // (self.size * self.size)
                        col = ((var - 1) % (self.size * self.size)) // self.size
                        num = ((var - 1) % self.size) + 1
                        if 0 <= row < self.size and 0 <= col < self.size:
                            solution[row][col] = num
                return solution
            return None
        except Exception as e:
            raise SudokuError(f"Error solving puzzle: {str(e)}")

    def solve(self):
        """Solve the Sudoku puzzle"""
        try:
            # Skip multiprocessing if in test mode
            if hasattr(self, "_testing"):
                return self._solve_task()

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
