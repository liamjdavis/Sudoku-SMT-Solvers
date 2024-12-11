# test_dfs_solver.py
import pytest
from sudoku_smt_solvers.benchmarks.sudoku_generator.dfs_solver import DFSSolver


@pytest.fixture
def solver():
    return DFSSolver(size=9)  # Using 9x9 grid for simpler testing


def test_initialization():
    solver = DFSSolver(size=25)
    assert solver.size == 25
    assert solver.box_size == 5
    assert len(solver.rows) == 25
    assert len(solver.cols) == 25
    assert len(solver.boxes) == 25


def test_get_subgrid_index():
    solver = DFSSolver(size=9)
    assert solver.get_subgrid_index(0, 0) == 0
    assert solver.get_subgrid_index(4, 4) == 4
    assert solver.get_subgrid_index(8, 8) == 8


def test_setup_board(solver):
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    grid[4][4] = 5

    solver.setup_board(grid)

    assert (0, 1) in solver.unfilled_positions
    assert 1 in solver.rows[0]
    assert 1 in solver.cols[0]
    assert 1 in solver.boxes[0]
    assert 5 in solver.rows[4]
    assert 5 in solver.cols[4]
    assert 5 in solver.boxes[4]


def test_get_valid_numbers(solver):
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    grid[0][1] = 2
    solver.setup_board(grid)

    valid_nums = solver.get_valid_numbers(0, 2)
    assert valid_nums == {3, 4, 5, 6, 7, 8, 9}


def test_solve_simple_puzzle(solver):
    grid = [
        [2, 9, 5, 7, 4, 3, 8, 6, 1],
        [4, 3, 1, 8, 6, 5, 9, 0, 0],
        [8, 7, 6, 1, 9, 2, 5, 4, 3],
        [3, 8, 7, 4, 5, 9, 2, 1, 6],
        [6, 1, 2, 3, 8, 7, 4, 9, 5],
        [5, 4, 9, 2, 1, 6, 7, 3, 8],
        [7, 6, 3, 5, 2, 4, 1, 8, 9],
        [9, 2, 8, 6, 7, 1, 3, 5, 4],
        [1, 5, 4, 9, 3, 8, 6, 0, 0],
    ]

    solutions = solver.solve(grid)
    assert len(solutions) == 2

    # Verify solutions
    expected_second_rows = [[4, 3, 1, 8, 6, 5, 9, 7, 2], [4, 3, 1, 8, 6, 5, 9, 2, 7]]
    assert solutions[0][1] in expected_second_rows

    expected_last_rows = [[1, 5, 4, 9, 3, 8, 6, 7, 2], [1, 5, 4, 9, 3, 8, 6, 2, 7]]
    assert solutions[1][-1] in expected_last_rows
