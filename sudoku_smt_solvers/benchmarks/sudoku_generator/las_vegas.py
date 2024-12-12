import random
import time
import csv
from dfs_solver import DFSSolver
from typing import List, Set, Tuple
from multiprocessing import Process, Queue


def solve_with_timeout(grid, solver, queue):
    start_time = time.time()
    solutions = solver.solve(grid)
    solve_time = time.time() - start_time
    queue.put((solutions, solve_time))


class LasVegasGenerator:
    def __init__(self, size: int = 25, givens: int = 80, timeout: int | None = 5):
        self.size = size
        self.givens = givens
        self.timeout = timeout
        self.all_positions = [
            (i, j) for i in range(self.size) for j in range(self.size)
        ]  # Create list of all possible positions
        self.rows = [set() for _ in range(self.size)]
        self.cols = [set() for _ in range(self.size)]
        self.boxes = [set() for _ in range(self.size)]
        self.solver = DFSSolver(size, find_all=False)

        box_size = int(self.size**0.5)
        self.box_lookup = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                self.box_lookup[i][j] = (i // box_size) * box_size + (j // box_size)

    def create_empty_grid(self) -> List[List[int]]:
        return [[0 for _ in range(self.size)] for _ in range(self.size)]

    def get_random_positions(self) -> Set[Tuple[int, int]]:
        # Use random.sample to select givens number of positions without replacement
        selected_positions = random.sample(self.all_positions, self.givens)

        return set(selected_positions)

    def is_valid_number(
        self, grid: List[List[int]], row: int, col: int, num: int
    ) -> bool:
        # Check if number exists in row, column or box using sets
        if (
            num in self.rows[row]
            or num in self.cols[col]
            or num in self.boxes[self.box_lookup[row][col]]
        ):
            return False
        return True

    def fill_random_positions(
        self, grid: List[List[int]], positions: Set[Tuple[int, int]]
    ) -> bool:
        # Clear existing sets
        self.rows = [set() for _ in range(self.size)]
        self.cols = [set() for _ in range(self.size)]
        self.boxes = [set() for _ in range(self.size)]

        positions_list = list(positions)
        random.shuffle(positions_list)

        def backtrack(pos_index: int) -> bool:
            if pos_index == len(positions_list):
                return True

            row, col = positions_list[pos_index]
            valid_numbers = [
                num
                for num in range(1, self.size + 1)
                if self.is_valid_number(grid, row, col, num)
            ]

            random.shuffle(valid_numbers)

            for num in valid_numbers:
                grid[row][col] = num
                self.rows[row].add(num)
                self.cols[col].add(num)
                self.boxes[self.box_lookup[row][col]].add(num)

                if backtrack(pos_index + 1):
                    return True

                grid[row][col] = 0
                self.rows[row].remove(num)
                self.cols[col].remove(num)
                self.boxes[self.box_lookup[row][col]].remove(num)

            return False

        return backtrack(0)

    def generate(self) -> List[List[int]]:
        attempts = 0

        while True:
            attempts += 1
            print(f"Attempt {attempts}...")
            attempt_start = time.time()

            grid = self.create_empty_grid()
            positions = self.get_random_positions()

            if not self.fill_random_positions(grid, positions):
                continue

            queue = Queue()
            process = Process(
                target=solve_with_timeout, args=(grid, self.solver, queue)
            )
            process.start()

            if self.timeout is not None:
                process.join(timeout=self.timeout)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    continue
            else:
                process.join()
                attempt_duration = time.time() - attempt_start
                print(f"Attempt duration: {attempt_duration:.2f} seconds")

            try:
                solution, solve_time = queue.get_nowait()
                if solution:
                    print(f"Found valid puzzle after {attempts} attempts")
                    print(f"Solving time: {solve_time:.2f} seconds")
                    return solution
            except Exception:  # Changed from queue.Empty to catch any queue errors
                continue


def print_grid(grid: List[List[int]]):
    size = len(grid)
    box_size = int(size**0.5)

    for i, row in enumerate(grid):
        if i > 0 and i % box_size == 0:
            print("-" * (size * 3 + box_size))

        for j, num in enumerate(row):
            if j > 0 and j % box_size == 0:
                print("|", end=" ")
            print(f"{num:2}", end=" ")
        print()


def save_grid_to_file(grid: List[List[int]], filename: str):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid)


if __name__ == "__main__":
    generator = LasVegasGenerator(timeout=300)
    puzzle = generator.generate()
    print("\nGenerated Puzzle:")
    print_grid(puzzle)
