import os
import json
import copy
from datetime import datetime
from typing import List, Tuple

from .las_vegas import LasVegasGenerator
from .hole_digger import HoleDigger


class SudokuGenerator:
    """Generates Sudoku puzzles and their solutions using Las Vegas algorithm.

    This class handles the complete puzzle generation process:
    1. Generates a complete valid solution using Las Vegas algorithm
    2. Creates holes in the solution to create the actual puzzle
    3. Saves both puzzle and solution with metadata

    Attributes:
        size (int): Size of the Sudoku grid (e.g., 25 for 25x25)
        givens (int): Number of initial filled positions for Las Vegas
        timeout (int): Maximum generation attempt time in seconds
        difficulty (str): Target difficulty level for hole creation
        puzzles_dir (str): Directory for storing generated puzzles
        solutions_dir (str): Directory for storing solutions

    Example:
        >>> generator = SudokuGenerator(size=9, difficulty="Hard")
        >>> puzzle, solution, puzzle_id = generator.generate()
    """

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
        """Generate a complete Sudoku puzzle and solution pair.

        Uses a two-step process:
        1. Las Vegas algorithm generates a complete valid solution
        2. Hole digger creates the puzzle by removing numbers

        Returns:
            tuple: Contains:
                - List[List[int]]: The puzzle grid with holes
                - List[List[int]]: The complete solution grid
                - str: Unique identifier for the puzzle/solution pair

        Note:
            The generated puzzle is guaranteed to have a unique solution
        """
        # Step 1: Generate complete solution using Las Vegas
        generator = LasVegasGenerator(self.size, self.givens, self.timeout)
        solution = generator.generate()

        # Step 2: Create holes using HoleDigger
        digger = HoleDigger(copy.deepcopy(solution), self.difficulty)
        puzzle = digger.dig_holes()

        # Generate unique identifier for this puzzle
        puzzle_id = self._generate_puzzle_id()

        # Save both puzzle and solution
        self._save_grid(puzzle, puzzle_id, is_puzzle=True)
        self._save_grid(solution, puzzle_id, is_puzzle=False)

        return puzzle, solution, puzzle_id

    def _generate_puzzle_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sudoku_{self.size}x{self.size}_{self.difficulty}_{timestamp}"

    def _save_grid(self, grid: List[List[int]], puzzle_id: str, is_puzzle: bool):
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
