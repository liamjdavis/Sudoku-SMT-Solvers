from typing import List


def get_var(row: int, col: int, num: int, size: int) -> int:
    """Convert Sudoku position and number to CNF variable."""
    # Validate inputs
    if not (0 <= row < size):
        raise ValueError(f"Row must be between 0 and {size-1}")
    if not (0 <= col < size):
        raise ValueError(f"Column must be between 0 and {size-1}")
    if not (1 <= num <= size):
        raise ValueError(f"Number must be between 1 and {size}")

    return row * size * size + col * size + num


def generate_cnf(size: int) -> List[List[int]]:
    """Generate CNF clauses for Sudoku rules."""
    # Validate size
    if size <= 0:
        raise ValueError("Size must be positive")

    # Check if size is a perfect square
    block_size = int(size**0.5)
    if block_size * block_size != size:
        raise ValueError("Size must be a perfect square")

    # Rest of the function remains the same
    clauses = []
    block_size = int(size**0.5)

    # Cell constraints
    for i in range(size):
        for j in range(size):
            clauses.append([get_var(i, j, k, size) for k in range(1, size + 1)])
            for k in range(1, size + 1):
                for l in range(k + 1, size + 1):
                    clauses.append([-get_var(i, j, k, size), -get_var(i, j, l, size)])

    # Row constraints
    for i in range(size):
        for k in range(1, size + 1):
            for j in range(size):
                for l in range(j + 1, size):
                    clauses.append([-get_var(i, j, k, size), -get_var(i, l, k, size)])

    # Column constraints
    for j in range(size):
        for k in range(1, size + 1):
            for i in range(size):
                for l in range(i + 1, size):
                    clauses.append([-get_var(i, j, k, size), -get_var(l, j, k, size)])

    # Block constraints
    for block_i in range(block_size):
        for block_j in range(block_size):
            for k in range(1, size + 1):
                for i in range(block_size):
                    for j in range(block_size):
                        for i2 in range(i, block_size):
                            for j2 in range(j + 1 if i2 == i else 0, block_size):
                                pos1 = (
                                    block_i * block_size + i,
                                    block_j * block_size + j,
                                )
                                pos2 = (
                                    block_i * block_size + i2,
                                    block_j * block_size + j2,
                                )
                                clauses.append(
                                    [-get_var(*pos1, k, size), -get_var(*pos2, k, size)]
                                )
    return clauses
