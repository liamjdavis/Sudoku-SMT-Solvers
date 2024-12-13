from typing import List, Optional
import time
import atexit
from cvc5 import Kind, Solver
from .sudoku_error import SudokuError


class CVC5Solver:
    def __init__(self, sudoku, timeout=120):
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")

        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self._validate_input(sudoku)
        self.sudoku = sudoku
        self.size = len(sudoku)
        self.timeout = timeout
        self.solver = None
        self.variables = None
        self.solve_time = 0
        self.propagated_clauses = 0
        self.start_time = None

    def _validate_input(self, sudoku):
        """Validate the input Sudoku grid."""
        for i, row in enumerate(sudoku):
            if not isinstance(row, list) or len(row) != 25:
                raise SudokuError(f"Invalid Sudoku puzzle: row {i} must have 25 cells")
            for j, val in enumerate(row):
                if not isinstance(val, int) or not (0 <= val <= 25):
                    raise SudokuError(
                        f"Invalid value at position ({i},{j}): must be between 0 and 25"
                    )

    def create_variables(self):
        """Set self.variables as a 2D list containing the CVC5 variables."""
        self.solver = Solver()
        self.solver.setOption("produce-models", "true")
        self.solver.setOption("incremental", "true")
        self.solver.setLogic("QF_LIA")  # Quantifier-Free Linear Integer Arithmetic

        integer_sort = self.solver.getIntegerSort()
        self.variables = [
            [self.solver.mkConst(integer_sort, f"x_{i}_{j}") for j in range(25)]
            for i in range(25)
        ]
        atexit.register(self.cleanup)

    def encode_rules(self):
        """Encode the Sudoku rules into the solver."""
        # Domain constraints: Ensure each variable is between 1 and 25
        for i in range(25):
            for j in range(25):
                self.solver.assertFormula(
                    self.solver.mkTerm(
                        Kind.AND,
                        self.solver.mkTerm(
                            Kind.LEQ, self.solver.mkInteger(1), self.variables[i][j]
                        ),
                        self.solver.mkTerm(
                            Kind.LEQ, self.variables[i][j], self.solver.mkInteger(25)
                        ),
                    )
                )

        # Row constraints: Each number appears exactly once in each row
        for i in range(25):
            self.solver.assertFormula(
                self.solver.mkTerm(
                    Kind.DISTINCT, *[self.variables[i][j] for j in range(25)]
                )
            )

        # Column constraints: Each number appears exactly once in each column
        for j in range(25):
            self.solver.assertFormula(
                self.solver.mkTerm(
                    Kind.DISTINCT, *[self.variables[i][j] for i in range(25)]
                )
            )

        # 5x5 subgrid constraints: Each number appears exactly once in each subgrid
        for block_row in range(0, 25, 5):
            for block_col in range(0, 25, 5):
                block_vars = [
                    self.variables[i][j]
                    for i in range(block_row, block_row + 5)
                    for j in range(block_col, block_col + 5)
                ]
                self.solver.assertFormula(
                    self.solver.mkTerm(Kind.DISTINCT, *block_vars)
                )

    def encode_puzzle(self):
        """Encode the initial Sudoku puzzle into the solver."""
        for i in range(25):
            for j in range(25):
                if self.sudoku[i][j] != 0:  # Pre-filled cell
                    self.solver.assertFormula(
                        self.solver.mkTerm(
                            Kind.EQUAL,
                            self.variables[i][j],
                            self.solver.mkInteger(self.sudoku[i][j]),
                        )
                    )

    def extract_solution(self):
        """Extract the solution from the CVC5 model."""
        solution = [[0 for _ in range(25)] for _ in range(25)]
        for i in range(25):
            for j in range(25):
                solution[i][j] = self.solver.getValue(
                    self.variables[i][j]
                ).getIntegerValue()
        return solution

    def _solve_task(self):
        """Helper method to encode and solve the puzzle."""
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()

        result = self.solver.checkSat()
        if result.isSat():
            return self.extract_solution()
        return None

    def cleanup(self):
        """Clean up solver resources."""
        if self.solver:
            self.solver = None

    def validate_solution(self, solution):
        if not solution:
            return False

        # Check dimensions
        if len(solution) != self.size or any(len(row) != self.size for row in solution):
            return False

        valid_nums = set(range(1, self.size + 1))

        # Check rows
        if any(set(row) != valid_nums for row in solution):
            return False

        # Check columns
        for col in range(self.size):
            if set(solution[row][col] for row in range(self.size)) != valid_nums:
                return False

        # Check 5x5 subgrids
        subgrid_size = 5
        for box_row in range(0, self.size, subgrid_size):
            for box_col in range(0, self.size, subgrid_size):
                numbers = set()
                for i in range(subgrid_size):
                    for j in range(subgrid_size):
                        numbers.add(solution[box_row + i][box_col + j])
                if numbers != valid_nums:
                    return False

        return True

    def solve(self):
        """Solve the Sudoku puzzle."""
        self.start_time = time.time()
        try:
            self.create_variables()
            self.encode_rules()
            self.encode_puzzle()

            result = self.solver.checkSat()
            current_time = time.time()

            if current_time - self.start_time > self.timeout:
                raise SudokuError("Solver timed out")

            if result.isSat():
                solution = self.extract_solution()
                self.solve_time = time.time() - self.start_time

                if self.validate_solution(solution):
                    return solution
            return None

        except Exception as e:
            if "timed out" in str(e).lower():
                raise SudokuError("Solver timed out")
            raise SudokuError(f"Critical solver error: {str(e)}")
