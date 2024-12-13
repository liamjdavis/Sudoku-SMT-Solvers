from typing import List, Optional
from pysat.solvers import Solver
from pysat.formula import CNF
from .sudoku_error import SudokuError
import time


class DPLLTSolver:
    def __init__(self, sudoku: List[List[int]], timeout: int = 120) -> None:
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")
        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = 25
        self.timeout = timeout
        self.cnf = CNF()  # CNF object to store Boolean clauses
        self.solver = Solver(name="glucose3")  # Low-level SAT solver
        self.theory_state = {}  # Store theory constraints dynamically
        self.decision_level = 0
        self.propagated_clauses = 0
        self.solve_time = 0
        self.start_time = None  # Timeout tracking

    def _count_clause(self) -> None:
        """Increment propagated clauses counter when adding a clause"""
        self.propagated_clauses += 1

    def add_sudoku_clauses(self) -> None:
        """Add Sudoku constraints as CNF clauses."""
        size = self.size
        block_size = int(size**0.5)

        def get_var(row, col, num):
            """Map (row, col, num) to a unique variable."""
            return row * size * size + col * size + num

        # At least one number in each cell
        for row in range(size):
            for col in range(size):
                self.cnf.append([get_var(row, col, num) for num in range(1, size + 1)])
                self._count_clause()

                # At most one number in each cell
                for num1 in range(1, size + 1):
                    for num2 in range(num1 + 1, size + 1):
                        self.cnf.append(
                            [-get_var(row, col, num1), -get_var(row, col, num2)]
                        )
                        self._count_clause()

        # Add row constraints
        for row in range(size):
            for num in range(1, size + 1):
                self.cnf.append([get_var(row, col, num) for col in range(size)])
                self._count_clause()

        # Add column constraints
        for col in range(size):
            for num in range(1, size + 1):
                self.cnf.append([get_var(row, col, num) for row in range(size)])
                self._count_clause()

        # Add block constraints
        for block_row in range(block_size):
            for block_col in range(block_size):
                for num in range(1, size + 1):
                    self.cnf.append(
                        [
                            get_var(
                                block_row * block_size + i,
                                block_col * block_size + j,
                                num,
                            )
                            for i in range(block_size)
                            for j in range(block_size)
                        ]
                    )
                    self._count_clause()

        # Add initial assignments from the puzzle
        for row in range(size):
            for col in range(size):
                if self.sudoku[row][col] != 0:
                    num = self.sudoku[row][col]
                    self.cnf.append([get_var(row, col, num)])
                    self._count_clause()

    def theory_propagation(self) -> Optional[List[int]]:
        """
        Theory propagation: Enforce Sudoku-specific rules dynamically.
        Returns a conflict clause if a theory conflict is detected.
        """
        block_size = int(self.size**0.5)

        def block_index(row, col):
            return (row // block_size) * block_size + (col // block_size)

        # Track constraints dynamically
        for row in range(self.size):
            for col in range(self.size):
                if self.sudoku[row][col] != 0:
                    num = self.sudoku[row][col]
                    # Check row, column, and block constraints
                    if num in self.theory_state.get((row, "row"), set()):
                        return [-self.get_var(row, col, num)]
                    if num in self.theory_state.get((col, "col"), set()):
                        return [-self.get_var(row, col, num)]
                    if num in self.theory_state.get(
                        (block_index(row, col), "block"), set()
                    ):
                        return [-self.get_var(row, col, num)]

                    # Add constraints to theory state
                    self.theory_state.setdefault((row, "row"), set()).add(num)
                    self.theory_state.setdefault((col, "col"), set()).add(num)
                    self.theory_state.setdefault(
                        (block_index(row, col), "block"), set()
                    ).add(num)
        return None

    def extract_solution(self, model: List[int]) -> List[List[int]]:
        """Convert SAT model to Sudoku grid."""
        solution = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for var in model:
            if var > 0:  # Only consider positive assignments
                var -= 1
                num = var % self.size + 1
                col = (var // self.size) % self.size
                row = var // (self.size * self.size)
                solution[row][col] = num
        return solution

    def validate_solution(self, solution: List[List[int]]) -> bool:
        """
        Validate the Sudoku solution.
        Ensures all rows, columns, and blocks contain unique numbers from 1 to size.
        """
        size = self.size
        block_size = int(size**0.5)

        # Validate rows
        for row in solution:
            if len(set(row)) != size or not all(1 <= num <= size for num in row):
                return False

        # Validate columns
        for col in range(size):
            column = [solution[row][col] for row in range(size)]
            if len(set(column)) != size:
                return False

        # Validate blocks
        for block_row in range(block_size):
            for block_col in range(block_size):
                block = [
                    solution[block_row * block_size + i][block_col * block_size + j]
                    for i in range(block_size)
                    for j in range(block_size)
                ]
                if len(set(block)) != size:
                    return False

        return True

    def solve(self) -> Optional[List[List[int]]]:
        """Solve the Sudoku puzzle using DPLL(T)."""
        self.start_time = time.time()
        self.add_sudoku_clauses()
        self.solver.append_formula(self.cnf.clauses)

        while self.solver.solve():
            # Check timeout
            if time.time() - self.start_time > self.timeout:
                self.solve_time = self.timeout
                raise SudokuError(f"Solving timed out after {self.timeout} seconds")

            # Perform theory propagation
            conflict_clause = self.theory_propagation()
            if conflict_clause:
                # Add conflict clause and continue solving
                self.solver.add_clause(conflict_clause)
                self._count_clause()
            else:
                # Extract and validate the solution
                model = self.solver.get_model()
                solution = self.extract_solution(model)
                if self.validate_solution(solution):
                    self.solve_time = time.time() - self.start_time

                    return solution
                else:
                    self.solve_time = time.time() - self.start_time
                    raise SudokuError("Invalid solution generated.")

        self.solve_time = time.time() - self.start_time
        # If UNSAT, return None
        return None
