from typing import List, Optional
import atexit
from cvc5 import Kind, Solver
from .sudoku_error import SudokuError


class CVC5Solver:
    """CVC5-based Sudoku solver using SMT encoding.

    Solves 25x25 Sudoku puzzles by encoding the problem as an SMT formula and using
    CVC5 to find a satisfying assignment.

    Attributes:
        sudoku (List[List[int]]): Input puzzle as 25x25 grid
        size (int): Grid size (always 25)
        solver (Solver): CVC5 solver instance
        variables (List[List[Term]]): SMT variables for each cell
        propagated_clauses (int): Counter for clause assertions

    Example:
        >>> puzzle = [[0 for _ in range(25)] for _ in range(25)]  # Empty puzzle
        >>> solver = CVC5Solver(puzzle)
        >>> solution = solver.solve()
    """

    def __init__(self, sudoku):
        """Initialize CVC5 Sudoku solver.

        Args:
            sudoku: 25x25 grid with values 0-25 (0 for empty cells)

        Raises:
            SudokuError: If puzzle format is invalid
        """
        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self._validate_input(sudoku)
        self.sudoku = sudoku
        self.size = len(sudoku)
        self.solver = None
        self.variables = None
        self.propagated_clauses = 0

    def _validate_input(self, sudoku):
        for i, row in enumerate(sudoku):
            if not isinstance(row, list) or len(row) != 25:
                raise SudokuError(f"Invalid Sudoku puzzle: row {i} must have 25 cells")
            for j, val in enumerate(row):
                if not isinstance(val, int) or not (0 <= val <= 25):
                    raise SudokuError(
                        f"Invalid value at position ({i},{j}): must be between 0 and 25"
                    )

    def _count_assertion(self):
        self.propagated_clauses += 1

    def create_variables(self):
        self.solver = Solver()

        # Configure CVC5 solver options
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
        # Domain constraints
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
                self._count_assertion()

        # Row constraints
        for i in range(25):
            self.solver.assertFormula(
                self.solver.mkTerm(
                    Kind.DISTINCT, *[self.variables[i][j] for j in range(25)]
                )
            )
            self._count_assertion()

        # Column constraints
        for j in range(25):
            self.solver.assertFormula(
                self.solver.mkTerm(
                    Kind.DISTINCT, *[self.variables[i][j] for i in range(25)]
                )
            )
            self._count_assertion()

        # 5x5 subgrid constraints
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
                self._count_assertion()

    def encode_puzzle(self):
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
                    self._count_assertion()

    def extract_solution(self):
        solution = [[0 for _ in range(25)] for _ in range(25)]
        for i in range(25):
            for j in range(25):
                solution[i][j] = self.solver.getValue(
                    self.variables[i][j]
                ).getIntegerValue()
        return solution

    def cleanup(self):
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
        """Solve the Sudoku puzzle using CVC5.

        Returns:
            Solved 25x25 grid if satisfiable, None if unsatisfiable

        Raises:
            Exception: If solver encounters an error

        Note:
            Always cleans up solver resources, even on failure
        """
        try:
            self.create_variables()
            self.encode_rules()
            self.encode_puzzle()

            result = self.solver.checkSat()

            if result.isSat():
                solution = self.extract_solution()
                if self.validate_solution(solution):
                    return solution
            return None

        except Exception as e:
            raise
        finally:
            self.cleanup()
