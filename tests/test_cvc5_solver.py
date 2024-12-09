import pytest
import threading
from unittest.mock import Mock, patch, create_autospec
from cvc5 import Kind, Solver
from sudoku_smt_solvers.solvers.cvc5_solver import CVC5Solver
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
    solver = CVC5Solver(valid_empty_grid)
    assert solver.sudoku == valid_empty_grid
    assert solver.timeout == 120


def test_init_invalid_timeout():
    with pytest.raises(SudokuError, match="Timeout must be positive"):
        CVC5Solver([[0] * 25 for _ in range(25)], timeout=0)


def test_init_none():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        CVC5Solver(None)


def test_init_invalid_grid_type():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        CVC5Solver("not a list")


def test_init_wrong_size():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        CVC5Solver([[0] * 25 for _ in range(24)])


def test_init_invalid_row():
    grid = [[0] * 25 for _ in range(25)]
    grid[0] = "invalid"
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: row 0 must have 25 cells"
    ):
        CVC5Solver(grid)


def test_init_invalid_value_type():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = "1"
    with pytest.raises(SudokuError, match="must be an integer"):
        CVC5Solver(grid)


def test_init_invalid_value_range():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = 26
    with pytest.raises(SudokuError, match="must be between 0 and 25"):
        CVC5Solver(grid)


# Variable Creation Tests
def test_create_variables_success(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    solver.create_variables()
    assert len(solver.variables) == 25
    assert all(len(row) == 25 for row in solver.variables)
    assert all(len(cell) == 25 for row in solver.variables for cell in row)


def test_create_variables_failure(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    with patch("cvc5.Solver") as mock_solver_class:
        mock_solver = mock_solver_class.return_value
        # Fix: Mock the correct method and add getBooleanSort mock
        mock_solver.getBooleanSort.return_value = Mock()
        mock_solver.mkConst.side_effect = Exception("CVC5 Error")
        solver.solver = mock_solver
        with pytest.raises(SudokuError, match="Failed to create CVC5 variables"):
            solver.create_variables()


# Encoding Tests
def test_encode_rules_uninitialized(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


def test_encode_puzzle_uninitialized(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_puzzle()


def test_encode_rules_failure(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    mock_solver = create_autospec(Solver)
    mock_solver.mkTerm.side_effect = Exception("CVC5 Error")
    solver.solver = mock_solver
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    with pytest.raises(SudokuError, match="Failed to encode cell constraint"):
        solver.encode_rules()


def test_encode_puzzle_failure(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    mock_solver = create_autospec(Solver)
    mock_solver.assertFormula.side_effect = Exception("CVC5 Error")
    solver.solver = mock_solver
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    with pytest.raises(SudokuError, match="Failed to encode initial value"):
        solver.encode_puzzle()


# Solving Tests
def test_solve_timeout(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid, timeout=0.1)
    # Don't set _testing flag here since we want to test actual timeout

    with pytest.raises(SudokuError, match=r"Solver timed out after 0.1 seconds"):
        solver.solve()


def test_solve_unknown(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    mock_solver = create_autospec(Solver)
    mock_solver.setOption.return_value = None
    mock_solver.setLogic.return_value = None
    mock_solver.checkSat.side_effect = Exception("Unknown error occurred")
    solver.solver = mock_solver

    with pytest.raises(SudokuError):
        solver.solve()


def test_solve_invalid_solution(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solver._testing = True

    # Create proper mock solver
    mock_solver = create_autospec(Solver)
    mock_solver.checkSat.return_value = Mock(isSat=lambda: True)

    # Create a proper mock for getValue that returns a callable
    mock_model = Mock()
    mock_model.getBooleanValue = Mock(return_value=True)
    mock_getValue = Mock(return_value=mock_model)
    mock_solver.getValue = mock_getValue

    # Set up the solver instance
    solver.solver = mock_solver
    solver.variables = [
        [[Mock() for _ in range(25)] for _ in range(25)] for _ in range(25)
    ]

    # Mock extract_solution to return a dummy solution
    dummy_solution = [[1 for _ in range(25)] for _ in range(25)]

    with (
        patch.object(solver, "_solve_task", return_value=mock_getValue),
        patch.object(solver, "extract_solution", return_value=dummy_solution),
        patch.object(solver, "_validate_solution", return_value=False),
    ):
        with pytest.raises(SudokuError, match="Generated solution is invalid"):
            result = solver.solve()
            if result:  # Ensure we reach validation for test mode
                solver._validate_solution(result)


# Solution Validation Tests
def test_validate_solution_none(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    assert not solver._validate_solution(None)


def test_validate_solution_wrong_dimensions(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    assert not solver._validate_solution([[1] * 24 for _ in range(25)])


def test_validate_solution_invalid_values(valid_empty_grid):
    solver = CVC5Solver(valid_empty_grid)
    invalid_solution = [[1] * 25 for _ in range(25)]
    invalid_solution[0][0] = 26
    assert not solver._validate_solution(invalid_solution)


def test_validate_solution_mismatch(valid_partial_grid):
    solver = CVC5Solver(valid_partial_grid)
    solution = [[2] * 25 for _ in range(25)]
    assert not solver._validate_solution(solution)


def test_solve_success(valid_partial_grid, solved_grid):
    solver = CVC5Solver(valid_partial_grid)
    solver._testing = True  # Add this line to enable test mode

    mock_solver = create_autospec(Solver)
    mock_result = Mock()
    mock_result.isSat.return_value = True
    mock_solver.checkSat.return_value = mock_result
    mock_solver.getValue = Mock()
    mock_solver.mkTerm.return_value = Mock()
    mock_solver.mkBoolean.return_value = Mock()

    mock_stats = Mock()
    mock_stats.get = Mock(return_value=0)
    mock_solver.getStats = Mock(return_value=mock_stats)
    solver.solver = mock_solver
    solver.variables = [
        [[Mock() for _ in range(25)] for _ in range(25)] for _ in range(25)
    ]

    with patch.object(solver, "create_variables"):
        with patch.object(solver, "encode_rules"):
            with patch.object(solver, "encode_puzzle"):
                with patch.object(solver, "extract_solution", return_value=solved_grid):
                    result = solver.solve()
                    assert result == solved_grid
                    assert solver.propagated_clauses == 0


def test_validate_input_no_list():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: input must be a list"
    ):
        solver = CVC5Solver([[0] * 25 for _ in range(25)])
        solver._validate_input(None)


def test_validate_input_inner_not_list():
    invalid_grid = [[0] * 25 for _ in range(25)]
    invalid_grid[5] = None
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: row 5 must have 25 cells"
    ):
        solver = CVC5Solver([[0] * 25 for _ in range(25)])
        solver._validate_input(invalid_grid)


def test_encode_rules_missing_solver():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    mock_solver = create_autospec(Solver)
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    solver.solver = None
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


def test_encode_rules_missing_variables():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    solver.solver = Solver()
    solver.variables = None
    with pytest.raises(SudokuError, match="Solver not initialized properly"):
        solver.encode_rules()


def test_encode_rules_value_constraint_error():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    mock_solver = create_autospec(Solver)
    mock_solver.mkTerm.side_effect = [None] * 10 + [Exception("Mock error")]
    solver.solver = mock_solver
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    with pytest.raises(SudokuError, match="Failed to encode value constraint"):
        solver.encode_rules()


def test_extract_solution_no_model():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    with pytest.raises(SudokuError, match="Invalid model: model cannot be None"):
        solver.extract_solution(None)


def test_validate_input_invalid_inner_value():
    invalid_grid = [[0] * 25 for _ in range(25)]
    invalid_grid[0][0] = "invalid"
    with pytest.raises(SudokuError, match="must be an integer"):
        solver = CVC5Solver([[0] * 25 for _ in range(25)])
        solver._validate_input(invalid_grid)


def test_encode_rules_value_constraint_multiple_failures():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    mock_solver = create_autospec(Solver)
    # First succeed, then fail on second constraint
    mock_solver.mkTerm.side_effect = [Mock(), Exception("Mock error")]
    mock_solver.assertFormula.side_effect = Exception("Assert error")
    solver.solver = mock_solver
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    with pytest.raises(SudokuError):
        solver.encode_rules()


def test_encode_puzzle_invalid_assert():
    solver = CVC5Solver([[1] * 25 for _ in range(25)])
    mock_solver = create_autospec(Solver)
    mock_solver.assertFormula.side_effect = Exception("Assert error")
    solver.solver = mock_solver
    solver.variables = [
        [
            [mock_solver.mkBoolean(f"x_{i}_{j}_{k}") for k in range(25)]
            for j in range(25)
        ]
        for i in range(25)
    ]
    with pytest.raises(SudokuError, match="Failed to encode initial value"):
        solver.encode_puzzle()


def test_extract_solution_missing_variables():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    solver.variables = None
    with pytest.raises(SudokuError, match="Variables not initialized"):
        solver.extract_solution(Mock())


def test_extract_solution_multiple_values():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    mock_model = Mock()
    mock_model.return_value.getBooleanValue.return_value = True
    solver.variables = [
        [[Mock() for _ in range(25)] for _ in range(25)] for _ in range(25)
    ]
    with pytest.raises(SudokuError, match="Multiple values assigned"):
        solver.extract_solution(mock_model)


def test_extract_solution_no_value():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    mock_model = Mock()
    mock_model.return_value.getBooleanValue.return_value = False
    solver.variables = [
        [[Mock() for _ in range(25)] for _ in range(25)] for _ in range(25)
    ]
    with pytest.raises(SudokuError, match="No value assigned to cell"):
        solver.extract_solution(mock_model)


def test_solve_create_variables_failure():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    # Set test mode
    solver._testing = True
    with patch.object(
        solver, "create_variables", side_effect=SudokuError("Mock error")
    ):
        with pytest.raises(SudokuError, match="Mock error"):
            solver.solve()


def test_solve_getStatistic_error():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    # Add this line to enable test mode
    solver._testing = True

    mock_solver = create_autospec(Solver)
    mock_result = Mock()
    mock_result.isSat.return_value = True
    mock_solver.checkSat.return_value = mock_result
    mock_solver.getValue = Mock()
    mock_solver.getStatistics.side_effect = Exception("Stats error")
    solver.solver = mock_solver
    solver.variables = [
        [[Mock() for _ in range(25)] for _ in range(25)] for _ in range(25)
    ]

    with patch.object(solver, "create_variables"):
        with patch.object(solver, "encode_rules"):
            with patch.object(solver, "encode_puzzle"):
                with patch.object(
                    solver, "extract_solution", return_value=[[1] * 25] * 25
                ):
                    with patch.object(solver, "_validate_solution", return_value=True):
                        result = solver.solve()
                        assert solver.propagated_clauses == 0


def test_init_empty_grid():
    with pytest.raises(
        SudokuError, match="Invalid Sudoku puzzle: must be a 25x25 grid"
    ):
        CVC5Solver([])  # Test with empty list
