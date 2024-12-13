import os
import json
from datetime import datetime
from typing import List, Tuple

from .las_vegas import LasVegasGenerator
from .hole_digger import HoleDigger


class SudokuGenerator:
    def __init__(
        self,
        size: int = 25,
        givens: int = 80,
        timeout: int = 5,
        difficulty: str = "Medium",
        puzzles_dir: str = "benchmarks/puzzles",
        solutions_dir: str = "benchmarks/solutions",
    ):
        """Initialize the Sudoku puzzle generator.

        Args:
            size: Grid size (default 25 for 25x25 grid)
            givens: Number of initial filled positions for Las Vegas
            timeout: Maximum time in seconds for generation attempts
            difficulty: Puzzle difficulty level for hole digger
            puzzles_dir: Directory to store generated puzzles
            solutions_dir: Directory to store solutions
        """
        self.size = size
        self.givens = givens
        self.timeout = timeout
        self.difficulty = difficulty
        self.puzzles_dir = puzzles_dir
        self.solutions_dir = solutions_dir

        # Create directories if they don't exist
        os.makedirs(puzzles_dir, exist_ok=True)
        os.makedirs(solutions_dir, exist_ok=True)

    def generate(self) -> Tuple[List[List[int]], List[List[int]], str]:
        """Generate a Sudoku puzzle and its solution.

        Returns:
            Tuple containing:
            - The puzzle (with holes)
            - The complete solution
            - The unique identifier for the puzzle/solution pair
        """
        # Step 1: Generate complete solution using Las Vegas
        generator = LasVegasGenerator(self.size, self.givens, self.timeout)
        solution = generator.generate()

        # Step 2: Create holes using HoleDigger
        digger = HoleDigger(solution, self.difficulty)
        puzzle = digger.dig_holes()

        # Generate unique identifier for this puzzle
        puzzle_id = self._generate_puzzle_id()

        # Save both puzzle and solution
        self._save_grid(puzzle, puzzle_id, is_puzzle=True)
        self._save_grid(solution, puzzle_id, is_puzzle=False)

        return puzzle, solution, puzzle_id

    def _generate_puzzle_id(self) -> str:
        """Generate a unique identifier for a puzzle."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sudoku_{self.size}x{self.size}_{self.difficulty}_{timestamp}"

    def _save_grid(self, grid: List[List[int]], puzzle_id: str, is_puzzle: bool):
        """Save a grid (puzzle or solution) to file.

        Args:
            grid: The grid to save
            puzzle_id: Unique identifier for the puzzle
            is_puzzle: True if saving puzzle, False if saving solution
        """
        directory = self.puzzles_dir if is_puzzle else self.solutions_dir
        filename = f"{puzzle_id}_{'puzzle' if is_puzzle else 'solution'}.json"
        filepath = os.path.join(directory, filename)

        metadata = {
            "size": self.size,
            "difficulty": self.difficulty,
            "givens": sum(cell != 0 for row in grid for cell in row),
            "type": "puzzle" if is_puzzle else "solution",
            "grid": grid,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
