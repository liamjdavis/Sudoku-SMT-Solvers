import pytest
from benchmarks.sudoku_generator.las_vegas import (
    LasVegasGenerator,
    print_grid,
    solve_with_timeout,
)
from unittest.mock import patch, MagicMock
from multiprocessing import Queue


@pytest.fixture
def generator():
    return LasVegasGenerator(size=9, givens=30, timeout=1)


@pytest.fixture
def small_generator():
    return LasVegasGenerator(size=4, givens=8, timeout=1)


def test_generator_initialization(generator):
    assert generator.size == 9
    assert generator.givens == 30
    assert generator.timeout == 1
    assert len(generator.all_positions) == 81
    assert len(generator.rows) == 9
    assert len(generator.cols) == 9
    assert len(generator.boxes) == 9


def test_create_empty_grid(generator):
    grid = generator.create_empty_grid()
    assert len(grid) == 9
    assert all(len(row) == 9 for row in grid)
    assert all(all(cell == 0 for cell in row) for row in grid)


def test_get_random_positions(generator):
    positions = generator.get_random_positions()
    assert len(positions) == 30
    assert all(isinstance(pos, tuple) for pos in positions)
    assert all(0 <= x < 9 and 0 <= y < 9 for x, y in positions)


def test_is_valid_number(generator):
    grid = generator.create_empty_grid()
    assert generator.is_valid_number(grid, 0, 0, 1) == True

    # Add a number and test validity
    generator.rows[0].add(1)
    assert generator.is_valid_number(grid, 0, 1, 1) == False
    generator.rows[0].remove(1)


def test_fill_random_positions(generator):
    grid = generator.create_empty_grid()
    positions = {(0, 0), (1, 1), (2, 2)}
    result = generator.fill_random_positions(grid, positions)
    assert result == True
    assert all(grid[i][i] != 0 for i in range(3))


def test_generate_small_puzzle(small_generator):
    puzzle = small_generator.generate()
    size = 4

    # Basic structure tests
    assert len(puzzle) == size
    assert all(len(row) == size for row in puzzle)
    assert any(any(cell != 0 for cell in row) for row in puzzle)

    # Row validation
    for row in puzzle:
        filled_numbers = [n for n in row if n != 0]
        assert all(1 <= n <= size for n in filled_numbers)
        assert len(filled_numbers) == len(set(filled_numbers))  # Check uniqueness

    # Column validation
    for col in range(size):
        filled_numbers = [
            puzzle[row][col] for row in range(size) if puzzle[row][col] != 0
        ]
        assert all(1 <= n <= size for n in filled_numbers)
        assert len(filled_numbers) == len(set(filled_numbers))

    # Box validation
    for box_row in range(0, size, 2):
        for box_col in range(0, size, 2):
            box_numbers = []
            for i in range(2):
                for j in range(2):
                    if puzzle[box_row + i][box_col + j] != 0:
                        box_numbers.append(puzzle[box_row + i][box_col + j])
            assert all(1 <= n <= size for n in box_numbers)
            assert len(box_numbers) == len(set(box_numbers))


@patch("benchmarks.sudoku_generator.las_vegas.Process")
def test_generate_with_timeout(mock_process, generator):
    mock_process.return_value.is_alive.return_value = False
    mock_queue = MagicMock()
    mock_queue.get_nowait.return_value = ([[1, 2, 3]], 0.1)

    with patch(
        "benchmarks.sudoku_generator.las_vegas.Queue",
        return_value=mock_queue,
    ):
        result = generator.generate()
        assert result == [[1, 2, 3]]


def test_solve_with_timeout():
    grid = [[1, 2], [3, 4]]
    mock_solver = MagicMock()
    mock_solver.solve.return_value = grid
    queue = Queue()

    solve_with_timeout(grid, mock_solver, queue)
    result, solve_time = queue.get()

    assert result == grid
    assert isinstance(solve_time, float)


def test_print_grid(capsys):
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print_grid(grid)
    captured = capsys.readouterr()
    assert captured.out != ""
    assert "1" in captured.out
    assert "9" in captured.out


def test_generator_edge_cases(generator):
    # Test with minimum givens
    min_generator = LasVegasGenerator(size=9, givens=1, timeout=1)
    assert min_generator.givens == 1

    # Test with maximum givens
    max_generator = LasVegasGenerator(size=9, givens=81, timeout=1)
    assert max_generator.givens == 81
