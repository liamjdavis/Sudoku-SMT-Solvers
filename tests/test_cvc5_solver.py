import pytest
import os
import json
from solvers.cvc5_solver import CVC5Solver
from solvers.utils.sudoku_error import SudokuError


@pytest.fixture
def valid_empty_grid():
    return [[0 for _ in range(25)] for _ in range(25)]


@pytest.fixture
def valid_partial_grid():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, "valid_partial_grid_puzzle.json")) as f:
        return json.load(f)


@pytest.fixture
def valid_partial_grid_solution():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, "valid_partial_grid_solution.json")) as f:
        return json.load(f)


@pytest.fixture
def solved_grid():
    # Create a valid solved 25x25 grid
    base = list(range(1, 26))
    grid = []
    for i in range(25):
        row = base[i:] + base[:i]
        grid.append(row)
    return grid


def test_init_valid(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    assert solver.size == 25


def test_init_invalid_grid_none():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        CVC5Solver(None)


def test_init_invalid_grid_size():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        CVC5Solver([[0] * 24])  # Wrong size


def test_validate_input_invalid_row(valid_empty_grid):
    invalid_grid = valid_empty_grid.copy()
    invalid_grid[0] = [0] * 24  # Wrong row length
    with pytest.raises(SudokuError, match="row 0 must have 25 cells"):
        CVC5Solver(invalid_grid)


def test_validate_input_invalid_value(valid_empty_grid):
    invalid_grid = valid_empty_grid.copy()
    invalid_grid[0][0] = 26  # Value out of range
    with pytest.raises(SudokuError, match="Invalid value at position"):
        CVC5Solver(invalid_grid)


def test_create_variables(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solver.create_variables()
    assert solver.solver is not None
    assert len(solver.variables) == 25
    assert len(solver.variables[0]) == 25


def test_encode(valid_partial_grid, valid_partial_grid_solution):
    solver = CVC5Solver(valid_partial_grid)
    solver.create_variables()
    solver.encode_rules()
    solver.encode_puzzle()
    # Verify the solver is still in a valid state
    assert solver.solver.checkSat().isSat()

    # Verify that the constraints for filled cells match the solution
    # Check first cell
    model = solver.solver.getValue(solver.variables[0][0])
    assert model.getIntegerValue() == valid_partial_grid_solution[0][0]

    # Check last cell
    model = solver.solver.getValue(solver.variables[24][24])
    assert model.getIntegerValue() == valid_partial_grid_solution[24][24]


def test_extract_solution(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solver.create_variables()
    solver.encode_rules()
    solver.encode_puzzle()
    assert solver.solver.checkSat().isSat()
    solution = solver.extract_solution()
    assert len(solution) == 25
    assert all(len(row) == 25 for row in solution)


def test_validate_solution_valid(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solution = solver.solve()
    assert solver.validate_solution(solution)

    # Verify solution maintains grid properties
    for row in solution:
        assert len(set(row)) == 25  # All numbers 1-25 appear once
        assert all(1 <= x <= 25 for x in row)  # All values in valid range


def test_validate_solution_invalid_dimensions():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    invalid_solution = [[1] * 24 for _ in range(25)]
    assert not solver.validate_solution(invalid_solution)


def test_solve_partial(valid_partial_grid, valid_partial_grid_solution):
    solver = CVC5Solver(valid_partial_grid)
    solution = solver.solve()

    # Verify solution exists and is valid
    assert solution is not None
    assert solver.validate_solution(solution)

    # Verify solution matches expected solution
    assert solution == valid_partial_grid_solution


def test_cleanup(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solver.create_variables()
    solver.cleanup()
    assert solver.solver is None


def test_solution_matches_expected(valid_partial_grid, valid_partial_grid_solution):
    solver = CVC5Solver(valid_partial_grid)
    solution = solver.solve()

    # Check entire solution matches
    for i in range(25):
        for j in range(25):
            assert (
                solution[i][j] == valid_partial_grid_solution[i][j]
            ), f"Mismatch at position ({i},{j}): {solution[i][j]} != {valid_partial_grid_solution[i][j]}"
