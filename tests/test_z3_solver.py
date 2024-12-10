import pytest
import multiprocessing
from unittest.mock import Mock, patch
from z3 import sat, unsat, Solver, Bool
from sudoku_smt_solvers.solvers.z3_solver import Z3Solver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_empty_grid():
    return [[0] * 25 for _ in range(25)]


@pytest.fixture
def valid_partial_grid():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = 1
    return grid


@pytest.fixture
def solved_grid():
    return [[(i * 25 + j) % 25 + 1 for j in range(25)] for i in range(25)]


# Initialization Tests
def test_init_valid(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    assert solver.sudoku == valid_empty_grid
    assert solver.timeout == 120


def test_init_invalid_timeout():
    with pytest.raises(SudokuError, match="Timeout must be positive"):
        Z3Solver([[0] * 25 for _ in range(25)], timeout=0)


def test_init_none():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        Z3Solver(None)


def test_init_invalid_grid_type():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        Z3Solver("not a list")


def test_init_wrong_size():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        Z3Solver([[0] * 25 for _ in range(24)])


def test_init_invalid_row():
    grid = [[0] * 25 for _ in range(25)]
    grid[0] = "invalid"
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: row 0 must have 25 cells"
    ):
        Z3Solver(grid)


def test_init_invalid_value_type():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = "1"
    with pytest.raises(SudokuError, match="must be an integer"):
        Z3Solver(grid)


def test_init_invalid_value_range():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = 26
    with pytest.raises(SudokuError, match="must be between 0 and 25"):
        Z3Solver(grid)


# Variable Creation Tests
def test_create_variables_success(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    solver.create_variables()
    assert len(solver.variables) == 25
    assert all(len(row) == 25 for row in solver.variables)
    assert all(len(cell) == 25 for row in solver.variables for cell in row)


@patch("sudoku_smt_solvers.solvers.z3_solver.Bool", side_effect=Exception("Z3 Error"))
def test_create_variables_failure(mock_bool, valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    with pytest.raises(SudokuError, match="Failed to create Z3 variables"):
        solver.create_variables()


# Encoding Tests
def test_encode_rules_uninitialized(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


def test_encode_puzzle_uninitialized(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_puzzle()


@patch.object(Solver, "add", side_effect=Exception("Z3 Error"))
def test_encode_rules_failure(mock_add, valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    solver.solver = Solver()
    solver.create_variables()
    with pytest.raises(SudokuError, match="Failed to encode cell constraint"):
        solver.encode_rules()


@patch.object(Solver, "add", side_effect=Exception("Z3 Error"))
def test_encode_puzzle_failure(mock_add, valid_partial_grid):
    solver = Z3Solver(valid_partial_grid)
    solver.solver = Solver()
    solver.create_variables()
    with pytest.raises(SudokuError, match="Failed to encode initial value"):
        solver.encode_puzzle()


# Solving Tests
def test_solve_timeout(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid, timeout=1)

    # Create mock for Pool.apply_async that raises TimeoutError
    mock_async_result = Mock()
    mock_async_result.get.side_effect = multiprocessing.TimeoutError()

    mock_apply_async = Mock(return_value=mock_async_result)
    mock_pool = Mock()
    mock_pool.apply_async = mock_apply_async

    # Mock Pool context manager
    mock_pool_instance = Mock(return_value=mock_pool)
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool)
    mock_pool_instance.__exit__ = Mock(return_value=None)

    with patch("multiprocessing.Pool", return_value=mock_pool_instance):
        with pytest.raises(SudokuError, match="Solver timed out"):
            solver.solve()


@patch.object(Solver, "check", return_value=unsat)
def test_solve_unsatisfiable(mock_check, valid_partial_grid):
    solver = Z3Solver(valid_partial_grid)
    solver._testing = True  # Enable test mode
    result = solver.solve()
    assert result is None


@patch.object(Solver, "check", return_value="unknown")
def test_solve_unknown(mock_check, valid_partial_grid):
    solver = Z3Solver(valid_partial_grid)
    solver._testing = True  # Enable test mode
    result = solver.solve()
    assert result is None


@patch.object(Solver, "model")
@patch.object(Solver, "check", return_value=sat)
def test_solve_invalid_solution(mock_check, mock_model, valid_partial_grid):
    # Create solver instance and enable test mode
    solver = Z3Solver(valid_partial_grid)
    solver._testing = True  # Enable test mode to skip multiprocessing

    # Mock solver behavior
    mock_model.return_value = Mock()

    # Test invalid solution path
    with patch.object(solver, "extract_solution", return_value=None):
        with patch.object(solver, "_validate_solution", return_value=False):
            with pytest.raises(SudokuError, match="Generated solution is invalid"):
                solver.solve()

    # Verify mocks were called
    mock_check.assert_called_once()
    mock_model.assert_called_once()


# Solution Validation Tests
def test_validate_solution_none(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    assert not solver._validate_solution(None)


def test_validate_solution_wrong_dimensions(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    assert not solver._validate_solution([[1] * 24 for _ in range(25)])


def test_validate_solution_invalid_values(valid_empty_grid):
    solver = Z3Solver(valid_empty_grid)
    invalid_solution = [[1] * 25 for _ in range(25)]
    invalid_solution[0][0] = 26
    assert not solver._validate_solution(invalid_solution)


def test_validate_solution_mismatch(valid_partial_grid):
    solver = Z3Solver(valid_partial_grid)
    solution = [[2] * 25 for _ in range(25)]
    assert not solver._validate_solution(solution)


@patch("z3.Solver")
def test_solve_success(mock_z3_solver, valid_partial_grid, solved_grid):
    mock_solver = Mock()
    mock_solver.check.return_value = sat

    # Mock statistics correctly
    mock_stats = Mock()
    mock_stats.get = Mock(return_value=0)  # Match default value
    mock_solver.statistics.return_value = mock_stats

    mock_solver.model.return_value = Mock()
    mock_z3_solver.return_value = mock_solver

    solver = Z3Solver(valid_partial_grid)
    solver._testing = True  # Enable test mode

    with patch.object(solver, "extract_solution", return_value=solved_grid):
        with patch.object(solver, "_validate_solution", return_value=True):
            result = solver.solve()
            assert result == solved_grid
            assert solver.propagated_clauses == 0


def test_validate_input_no_list():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: input must be a list"
    ):
        solver = Z3Solver([[0] * 25 for _ in range(25)])
        solver._validate_input(None)


def test_validate_input_inner_not_list():
    invalid_grid = [[0] * 25 for _ in range(25)]
    invalid_grid[5] = None
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: row 5 must have 25 cells"
    ):
        solver = Z3Solver([[0] * 25 for _ in range(25)])
        solver._validate_input(invalid_grid)


def test_encode_rules_missing_solver():
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    solver.variables = [
        [[Bool(f"x_{i}_{j}_{k}") for k in range(25)] for j in range(25)]
        for i in range(25)
    ]
    solver.solver = None
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


def test_encode_rules_missing_variables():
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    solver.solver = Solver()
    solver.variables = None
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


@patch.object(Solver, "add")
def test_encode_rules_value_constraint_error(mock_add):
    mock_add.side_effect = [None] * 10 + [Exception("Mock error")]
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    solver.solver = Solver()
    solver.create_variables()
    with pytest.raises(SudokuError, match="Failed to encode value constraint"):
        solver.encode_rules()


def test_extract_solution_no_model():
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    with pytest.raises(SudokuError, match="Invalid model: model cannot be None"):
        solver.extract_solution(None)


def test_extract_solution_no_variables():
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    mock_model = Mock()
    with pytest.raises(SudokuError, match="Variables not initialized"):
        solver.extract_solution(mock_model)


@patch("z3.Model")
def test_extract_solution_multiple_values(mock_model):
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    solver.create_variables()
    mock_model.evaluate.return_value = True
    with pytest.raises(SudokuError, match="Multiple values assigned to cell"):
        solver.extract_solution(mock_model)


@patch("z3.Model")
def test_extract_solution_no_value(mock_model):
    solver = Z3Solver([[0] * 25 for _ in range(25)])
    solver.create_variables()
    mock_model.evaluate.return_value = False
    with pytest.raises(SudokuError, match="No value assigned to cell"):
        solver.extract_solution(mock_model)
