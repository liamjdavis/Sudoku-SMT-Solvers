import random
from typing import List, Tuple
from .dfs_solver import DFSSolver

from .las_vegas import LasVegasGenerator

# A mapping from difficulty to the range of givens
difficulty_givensRange_mapping = {
    "Extremely Easy": [382, float("inf")],  # More than 386 given cells
    "Easy": [274, 381],
    "Medium": [243, 273],
    "Difficult": [212, 242],
    "Evil": [166, 211],
}

# A mapping from difficulty to the lower bound of givens in each row and column
difficulty_lower_bound_mapping = {
    "Extremely Easy": 14,
    "Easy": 11,
    "Medium": 7,
    "Difficult": 4,
    "Evil": 0,
}

digging_sequence_mapping = {
    "Extremely Easy": "random",
    "Easy": "random",
    "Medium": "Jumping one cell",
    "Difficult": "Wandering along 'S'",
    "Evil": "Left to right, top to bottom",
}


class HoleDigger:
    def __init__(self, puzzle: List[List[int]], difficulty: str):
        self.puzzle = puzzle
        self.size = len(puzzle)
        self.difficulty = difficulty

        self.target_range = difficulty_givensRange_mapping[difficulty]
        self.num_givens = (
            random.randint(*self.target_range)
            if self.target_range[1] != float("inf")
            else random.randint(self.target_range[0], self.size**2)
        )

        self.lower_bound = difficulty_lower_bound_mapping[difficulty]
        self.pattern = digging_sequence_mapping[difficulty]

        self.can_be_dug = {(i, j) for i in range(self.size) for j in range(self.size)}
        self.remaining_cells = self.size * self.size

    def get_digging_sequence(self) -> List[Tuple[int, int]]:
        if self.pattern == "random":
            return self._random_pattern()
        elif self.pattern == "Jumping one cell":
            return self._jumping_pattern()
        elif self.pattern == "Wandering along 'S'":
            return self._s_pattern()
        else:  # "Left to right, top to bottom"
            return self._linear_pattern()

    def _random_pattern(self) -> List[Tuple[int, int]]:
        cells = list(self.can_be_dug)
        random.shuffle(cells)
        return cells

    def _jumping_pattern(self) -> List[Tuple[int, int]]:
        sequence = []
        # Forward pass - only on even rows (0, 2, 4...)
        for i in range(0, self.size, 2):
            for j in range(0, self.size, 2):
                sequence.append((i, j))

        # Backward pass - only on odd rows (1, 3, 5...)
        for i in range(1, self.size, 2):
            for j in range(self.size - 2, -1, -2):
                sequence.append((i, j))

        return sequence

    def _s_pattern(self) -> List[Tuple[int, int]]:
        sequence = []
        for i in range(self.size):
            row = range(self.size) if i % 2 == 0 else range(self.size - 1, -1, -1)
            for j in row:
                sequence.append((i, j))
        return sequence

    def _linear_pattern(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def pass_restrictions(self, row: int, col: int) -> bool:
        # Skip if already dug
        if self.puzzle[row][col] == 0:
            return False

        # Check if digging would violate minimum remaining cells
        if self.remaining_cells - 1 < self.num_givens:
            return False

        # Check row constraint
        row_remaining = sum(cell != 0 for cell in self.puzzle[row])
        if row_remaining - 1 < self.lower_bound:
            return False

        # Check column constraint
        col_remaining = sum(self.puzzle[i][col] != 0 for i in range(self.size))
        if col_remaining - 1 < self.lower_bound:
            return False

        return True

    def has_unique_solution(self) -> bool:
        solver = DFSSolver(size=self.size, solutions_limit=2)
        solutions = solver.solve(self.puzzle)
        return len(solutions) == 1

    def dig_holes(self) -> List[List[int]]:
        digging_sequence = self.get_digging_sequence()
        cells_to_check = set(digging_sequence)

        while cells_to_check:
            row, col = next(iter(cells_to_check))

            if not self.pass_restrictions(row, col):
                cells_to_check.remove((row, col))
                continue

            # Save current value in case we need to restore it
            current_value = self.puzzle[row][col]
            self.puzzle[row][col] = 0

            if not self.has_unique_solution():
                # Restore the value if digging creates multiple solutions
                self.puzzle[row][col] = current_value
                self.can_be_dug.remove((row, col))
            else:
                # Update remaining cells if digging is successful
                self.remaining_cells -= 1

            cells_to_check.remove((row, col))

        return self.puzzle
