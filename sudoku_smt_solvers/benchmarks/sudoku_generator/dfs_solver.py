from math import sqrt


class DFSSolver:
    def __init__(self, size=25):
        """Initialize solver with configurable grid size"""
        self.size = size  # Total grid size (default 25x25)
        self.box_size = int(sqrt(size))  # Size of each sub-box (default 5x5)
        self.rows = [set() for _ in range(self.size)]
        self.cols = [set() for _ in range(self.size)]
        self.boxes = [set() for _ in range(self.size)]
        self.unfilled_positions = []  # Track positions of empty cells

    def get_subgrid_index(self, row, col):
        """Calculate which subgrid (box) a cell belongs to"""
        return (row // self.box_size) * self.box_size + (col // self.box_size)

    def setup_board(self, grid):
        """Set up initial board state"""
        self.unfilled_positions.clear()
        for i in range(self.size):
            for j in range(self.size):
                num = grid[i][j]
                if num == 0:  # Empty cell
                    self.unfilled_positions.append((i, j))
                else:  # Pre-filled number
                    self.rows[i].add(num)
                    self.cols[j].add(num)
                    self.boxes[self.get_subgrid_index(i, j)].add(num)

    def get_valid_numbers(self, row, col):
        """Get valid numbers for a cell using set operations"""
        used = (
            self.rows[row]
            | self.cols[col]
            | self.boxes[self.get_subgrid_index(row, col)]
        )
        return set(range(1, self.size + 1)) - used

    def solve(self, grid):
        """Main solving function using backtracking"""
        self.setup_board(grid)
        solutions = []

        def search():
            """Depth-first search implementation"""
            if not self.unfilled_positions:  # Found a solution
                solutions.append([row[:] for row in grid])
                return

            # Select cell with minimum valid numbers to reduce branching
            min_candidates = float("inf")
            min_pos = None
            min_idx = None

            for idx, (row, col) in enumerate(self.unfilled_positions):
                valid_numbers = self.get_valid_numbers(row, col)
                if len(valid_numbers) < min_candidates:
                    min_candidates = len(valid_numbers)
                    min_pos = (row, col)
                    min_idx = idx
                    if min_candidates == 0:  # No valid numbers available
                        return

            row, col = min_pos
            self.unfilled_positions.pop(min_idx)

            for num in self.get_valid_numbers(row, col):
                # Place number and update board state
                grid[row][col] = num
                self.rows[row].add(num)
                self.cols[col].add(num)
                self.boxes[self.get_subgrid_index(row, col)].add(num)

                search()

                # Backtrack: remove number and restore board state
                grid[row][col] = 0
                self.rows[row].remove(num)
                self.cols[col].remove(num)
                self.boxes[self.get_subgrid_index(row, col)].remove(num)

            self.unfilled_positions.insert(min_idx, (row, col))

        search()
        return solutions
