from z3 import Solver, Bool, Int, And, Or, Not, Implies, Distinct, sat, unsat
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError
import time


class Z3Solver:
    def __init__(self, sudoku, timeout=120):
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")

        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = len(sudoku)
        self.timeout = timeout
        self.solver = None
        self.variables = None
        self.solve_time = 0
        self.propagated_clauses = 0
        self.start_time = None

    def create_variables(self):
        """
        Set self.variables as a 3D list containing the Z3 variables.
        self.variables[i][j][k] is true if cell i,j contains the value k+1.
        """
        self.variables = [
            [Int(f"x_{i}_{j}") for j in range(self.size)] for i in range(self.size)
        ]

    def encode_rules(self):
        """Encode Sudoku rules using Int variables and Distinct"""
        # Cell range constraints
        cell_constraints = []
        for i in range(self.size):
            for j in range(self.size):
                cell_constraints.append(1 <= self.variables[i][j])
                cell_constraints.append(self.variables[i][j] <= 25)
        self.solver.add(cell_constraints)

        # Row constraints unchanged
        row_constraints = [Distinct(self.variables[i]) for i in range(self.size)]
        self.solver.add(row_constraints)

        # Column constraints unchanged
        col_constraints = [
            Distinct([self.variables[i][j] for i in range(self.size)])
            for j in range(self.size)
        ]
        self.solver.add(col_constraints)

        # Box constraints unchanged
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

    def encode_puzzle(self):
        """Encode initial values directly"""
        initial_values = []
        for i in range(self.size):
            for j in range(self.size):
                if self.sudoku[i][j] != 0:
                    initial_values.append(self.variables[i][j] == self.sudoku[i][j])
        self.solver.add(initial_values)

    def extract_solution(self, model):
        """
        Extract solution from model.
        """
        return [
            [model.evaluate(self.variables[i][j]).as_long() for j in range(self.size)]
            for i in range(self.size)
        ]

    def validate_solution(self, solution):
        """
        Validate if the solution meets Sudoku rules.
        Returns True if valid, False otherwise.
        """
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
        """
        Solve the Sudoku puzzle.
        """
        self.start_time = time.time()
        self.solver = Solver()
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()

        # Set timeout
        self.solver.set("timeout", self.timeout * 1000)  # Z3 timeout is in milliseconds

        result = self.solver.check()
        current_time = time.time()

        # Check timeout
        if current_time - self.start_time > self.timeout:
            raise SudokuError("Solver timed out")

        if result == sat:
            model = self.solver.model()
            solution = self.extract_solution(model)
            self.solve_time = time.time() - self.start_time

            if self.validate_solution(solution):
                print(f"Solution found in {self.solve_time:.2f} seconds")
                return solution

        return None
