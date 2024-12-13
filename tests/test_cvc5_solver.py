import pytest
from sudoku_smt_solvers.solvers.cvc5_solver import CVC5Solver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_empty_grid():
    return [[0 for _ in range(25)] for _ in range(25)]


@pytest.fixture
def valid_partial_grid():
    grid = [[0 for _ in range(25)] for _ in range(25)]
    grid[0][0] = 1
    grid[24][24] = 25
    return grid


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
    assert solver.timeout == 120


def test_init_invalid_timeout():
    with pytest.raises(SudokuError, match="Timeout must be positive"):
        CVC5Solver([[]], timeout=0)


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


def test_encode_rules(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solver.create_variables()
    solver.encode_rules()
    # Verify the solver is still in a valid state
    assert solver.solver.checkSat().isSat()


def test_encode_puzzle(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solver.create_variables()
    solver.encode_rules()
    solver.encode_puzzle()
    # Verify the solver is still in a valid state
    assert solver.solver.checkSat().isSat()


def test_extract_solution(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solver.create_variables()
    solver.encode_rules()
    solver.encode_puzzle()
    assert solver.solver.checkSat().isSat()
    solution = solver.extract_solution()
    assert len(solution) == 25
    assert all(len(row) == 25 for row in solution)


def test_validate_solution_valid(solved_grid):
    solver = CVC5Solver(solved_grid)
    assert solver.validate_solution(solved_grid)


def test_validate_solution_invalid_none():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    assert not solver.validate_solution(None)


def test_validate_solution_invalid_dimensions():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    invalid_solution = [[1] * 24 for _ in range(25)]
    assert not solver.validate_solution(invalid_solution)


def test_solve_empty(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solution = solver.solve()
    assert solution is not None
    assert solver.validate_solution(solution)


def test_solve_partial(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solution = solver.solve()
    assert solution is not None
    assert solver.validate_solution(solution)
    assert solution[0][0] == 1
    assert solution[24][24] == 25


def test_solve_timeout():
    solver = CVC5Solver([[0] * 25 for _ in range(25)], timeout=0.001)
    with pytest.raises(SudokuError, match="Solver timed out"):
        solver.solve()


def test_cleanup(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solver.create_variables()
    solver.cleanup()
    assert solver.solver is None
