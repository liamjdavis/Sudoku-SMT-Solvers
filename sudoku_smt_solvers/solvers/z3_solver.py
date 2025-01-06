from z3 import Solver, Int, Distinct, sat
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


class Z3Solver:
    """Z3-based SMT solver for Sudoku puzzles.

    Uses integer variables and distinct constraints to encode Sudoku rules.
    Tracks constraint propagation for performance analysis.

    Attributes:
        sudoku (List[List[int]]): Input puzzle as 25x25 grid
        size (int): Grid size (25)
        solver (z3.Solver): Z3 solver instance
        variables (List[List[z3.Int]]): SMT variables for grid
        propagated_clauses (int): Counter for constraint additions

    Example:
        >>> puzzle = [[0 for _ in range(25)] for _ in range(25)]
        >>> solver = Z3Solver(puzzle)
        >>> solution = solver.solve()
    """

    def __init__(self, sudoku):
        """Initialize Z3 Sudoku solver.

        Args:
            sudoku: 25x25 grid with values 0-25 (0 for empty cells)

        Raises:
            SudokuError: If puzzle format is invalid
        """
        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = len(sudoku)
        self.solver = None
        self.variables = None
        self.propagated_clauses = 0

    def create_variables(self):
        self.variables = [
            [Int(f"x_{i}_{j}") for j in range(self.size)] for i in range(self.size)
        ]

    def _count_clause(self):
        self.propagated_clauses += 1

    def encode_rules(self):
        # Cell range constraints
        cell_constraints = []
        for i in range(self.size):
            for j in range(self.size):
                cell_constraints.append(1 <= self.variables[i][j])
                cell_constraints.append(self.variables[i][j] <= 25)
                self._count_clause()
                self._count_clause()
        self.solver.add(cell_constraints)

        # Row constraints
        row_constraints = [Distinct(self.variables[i]) for i in range(self.size)]
        self.solver.add(row_constraints)
        for _ in range(self.size):
            self._count_clause()

        # Column constraints
        col_constraints = [
            Distinct([self.variables[i][j] for i in range(self.size)])
            for j in range(self.size)
        ]
        self.solver.add(col_constraints)
        for _ in range(self.size):
            self._count_clause()

        # Box constraints
        box_constraints = [
            Distinct(
                [
                    self.variables[5 * box_i + i][5 * box_j + j]
                    for i in range(5)
                    for j in range(5)
                ]
            )
            for box_i in range(5)
            for box_j in range(5)
        ]
        self.solver.add(box_constraints)
        for _ in range(25):
            self._count_clause()

    def encode_puzzle(self):
        initial_values = []
        for i in range(self.size):
            for j in range(self.size):
                if self.sudoku[i][j] != 0:
                    initial_values.append(self.variables[i][j] == self.sudoku[i][j])
                    self._count_clause()
        self.solver.add(initial_values)

    def extract_solution(self, model):
        return [
            [model.evaluate(self.variables[i][j]).as_long() for j in range(self.size)]
            for i in range(self.size)
        ]

    def validate_solution(self, solution):
        # Check range
        for row in solution:
            if not all(1 <= num <= 25 for num in row):
                return False

        # Check rows
        for row in solution:
            if len(set(row)) != self.size:
                return False

        # Check columns
        for j in range(self.size):
            col = [solution[i][j] for i in range(self.size)]
            if len(set(col)) != self.size:
                return False

        # Check boxes
        for box_i in range(5):
            for box_j in range(5):
                box = [
                    solution[5 * box_i + i][5 * box_j + j]
                    for i in range(5)
                    for j in range(5)
                ]
                if len(set(box)) != self.size:
                    return False

        return True

    def solve(self):
        """Solve Sudoku using Z3 SMT solver.

        Returns:
            Solved 25x25 grid if satisfiable, None if unsatisfiable

        Note:
            Validates solution before returning to ensure correctness
        """
        self.solver = Solver()
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()

        result = self.solver.check()

        if result == sat:
            model = self.solver.model()
            solution = self.extract_solution(model)

            if self.validate_solution(solution):
                return solution

        return None
