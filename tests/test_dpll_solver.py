import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from sudoku_smt_solvers.solvers.dpll_solver import DPLLSolver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError
from sudoku_smt_solvers.solvers.utils import generate_cnf, get_var


@pytest.fixture
def valid_25x25_puzzle():
    # Create a 25x25 puzzle with some initial values
    puzzle = [[0] * 25 for _ in range(25)]
    puzzle[0][0] = 1  # Add some known values
    puzzle[1][1] = 2
    return puzzle


@pytest.fixture
def dpll_solver(valid_25x25_puzzle):
    return DPLLSolver(valid_25x25_puzzle)


def test_initialization(valid_25x25_puzzle):
    solver = DPLLSolver(valid_25x25_puzzle)
    assert solver.sudoku == valid_25x25_puzzle
    assert solver.timeout == 120
    assert isinstance(solver.clauses, list)
    assert isinstance(solver.assignments, dict)


def test_initialization_invalid_puzzle():
    with pytest.raises(SudokuError):
        DPLLSolver(None)


def test_find_unit_clause(dpll_solver):
    dpll_solver.clauses = [[1, 2], [3], [-4, 5]]
    dpll_solver.assignments = {2: True}
    unit = dpll_solver.find_unit_clause()
    assert unit == 3


def test_find_pure_literal(dpll_solver):
    dpll_solver.clauses = [[2, 3], [2, 4], [1, 4]]  # Changed clauses to make 4 pure
    dpll_solver.assignments = {}
    pure = dpll_solver.find_pure_literal()
    assert pure in {2}


def test_find_pure_literal_positive(dpll_solver):
    dpll_solver.clauses = [[1, 2], [-2, 3], [1, 4]]
    dpll_solver.assignments = {}
    pure = dpll_solver.find_pure_literal()
    assert pure in {1, 3, 4}


def test_find_pure_literal_negative(dpll_solver):
    dpll_solver.clauses = [[-1, -2, 3], [-2, -3], [-1, -4]]
    dpll_solver.assignments = {}
    pure = dpll_solver.find_pure_literal()
    assert pure in {-1, -3, -4}


def test_find_pure_literal_skip_assigned(dpll_solver):
    dpll_solver.clauses = [[1, 2], [2, 3], [1, 4]]
    dpll_solver.assignments = {2: True}
    pure = dpll_solver.find_pure_literal()
    assert pure in {1, 3, 4}


def test_update_clauses_valid(dpll_solver):
    dpll_solver.clauses = [[1, 2], [-1, 3], [2, 3]]
    assert dpll_solver.update_clauses(1)
    assert len(dpll_solver.clauses) < 3


def test_update_clauses_invalid(dpll_solver):
    dpll_solver.clauses = [[1], [-1]]
    assert not dpll_solver.update_clauses(1)


def test_dpll_satisfiable(dpll_solver):
    dpll_solver.clauses = [[1, 2], [-1, 2]]
    assert dpll_solver.dpll()
    assert 2 in dpll_solver.assignments


def test_dpll_unsatisfiable(dpll_solver):
    dpll_solver.clauses = [[1], [-1]]
    assert not dpll_solver.dpll()


def test_solve_valid(dpll_solver):
    # Setup test puzzle with correct size
    dpll_solver.sudoku = [[0] * 25 for _ in range(25)]
    dpll_solver.sudoku[0][0] = 1
    dpll_solver.sudoku[1][1] = 2
    dpll_solver.size = 25

    # Mock multiprocessing
    mock_pool = MagicMock()
    mock_async_result = MagicMock()
    mock_async_result.get.return_value = dpll_solver.sudoku
    mock_pool.__enter__.return_value.apply_async.return_value = mock_async_result

    with patch("multiprocessing.Pool", return_value=mock_pool):
        with patch("sudoku_smt_solvers.solvers.utils.get_var") as mock_get_var:
            mock_get_var.side_effect = (
                lambda row, col, num, size: row * size * size + col * size + num
            )

            # Generate minimal clauses
            dpll_solver.clauses = [
                [1],  # (0,0) = 1
                [8],  # (1,1) = 2
            ]
            dpll_solver.assignments = {}

            solution = dpll_solver.solve()

            assert solution is not None
            assert solution[0][0] == 1
            assert solution[1][1] == 2

            # Verify timeout was used
            mock_async_result.get.assert_called_once_with(timeout=120)


def test_solve_timeout(dpll_solver):
    # Mock the multiprocessing Pool
    with patch("multiprocessing.Pool") as mock_pool:
        # Configure mock pool instance
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance

        # Configure async result to raise TimeoutError
        mock_async = Mock()
        mock_async.get.side_effect = multiprocessing.TimeoutError()
        mock_pool_instance.apply_async.return_value = mock_async

        # Test timeout handling
        with pytest.raises(SudokuError) as exc_info:
            dpll_solver.solve()

        # Verify error message
        assert f"Solver timed out after {dpll_solver.timeout} seconds" in str(
            exc_info.value
        )

        # Verify pool was used correctly
        mock_pool_instance.apply_async.assert_called_once_with(dpll_solver._solve_task)
        mock_async.get.assert_called_once_with(timeout=dpll_solver.timeout)


def test_solve_invalid_puzzle():
    invalid_puzzle = "not a valid puzzle"
    with pytest.raises(SudokuError):
        solver = DPLLSolver(invalid_puzzle)
        solver.solve()


@pytest.fixture
def valid_25x25_puzzle():
    # Create a 25x25 puzzle with some initial values
    puzzle = [[0] * 25 for _ in range(25)]
    puzzle[0][0] = 1  # Add some known values
    puzzle[1][1] = 2
    return puzzle


@pytest.fixture
def dpll_solver(valid_25x25_puzzle):
    return DPLLSolver(valid_25x25_puzzle)


def test_dpll_pure_literal_elimination(dpll_solver):
    dpll_solver.clauses = [[1, 2], [2, 3], [4]]
    dpll_solver.assignments = {}
    assert dpll_solver.dpll()
    assert 4 in dpll_solver.assignments


def test_dpll_choose_variable(dpll_solver):
    dpll_solver.clauses = [[1, 2], [-1, 3], [2, 3]]
    dpll_solver.assignments = {}
    assert dpll_solver.dpll()
    assert 1 in dpll_solver.assignments or 2 in dpll_solver.assignments


def test_dpll_try_true(dpll_solver):
    dpll_solver.clauses = [[1, 2], [-1, 3], [2, 3]]
    dpll_solver.assignments = {}
    assert dpll_solver.dpll()
    assert 1 in dpll_solver.assignments or 2 in dpll_solver.assignments


def test_dpll_try_false(dpll_solver):
    dpll_solver.clauses = [[1, -2], [-1, 2], [2, 3]]
    dpll_solver.assignments = {}
    assert dpll_solver.dpll()
    assert 2 in dpll_solver.assignments


def test_solve_dpll_exception(dpll_solver):
    # Enable test mode
    dpll_solver._testing = True

    # Patch the dpll method of the specific instance rather than the class
    dpll_solver.dpll = Mock(side_effect=Exception("Mocked DPLL error"))

    with pytest.raises(SudokuError, match="Error solving puzzle: Mocked DPLL error"):
        dpll_solver.solve()


def test_dpll_branching_strategy(dpll_solver):
    # Setup a formula that requires branching
    dpll_solver.clauses = [[1, 2, 3], [-1, -2], [-1, -3], [-2, -3]]
    dpll_solver.assignments = {}

    # Track assignments tried
    assignments_tried = []

    # Store original method
    original_update = dpll_solver.update_clauses

    def mock_update_clauses(assignment):
        assignments_tried.append(assignment)
        if assignment == 2:  # When trying var2=True
            dpll_solver.assignments.clear()
            return False  # Force backtracking
        return original_update(assignment)  # Normal behavior otherwise

    try:
        dpll_solver.update_clauses = mock_update_clauses
        result = dpll_solver.dpll()

        # Verify results
        assert result == True  # Solution should be found
        assert -2 in assignments_tried  # Tried var2=False
        assert len(assignments_tried) >= 2  # At least tried both values for var2
        assert dpll_solver.assignments[1] == True  # var1 must be True in solution
        assert dpll_solver.assignments[2] == False  # var2 must be False in solution
        assert dpll_solver.assignments[3] == False  # var3 must be False in solution

    finally:
        dpll_solver.update_clauses = original_update


def test_solution_reconstruction(dpll_solver):
    # Enable test mode
    dpll_solver._testing = True

    # Setup test conditions
    dpll_solver.size = 25
    dpll_solver.assignments = {
        # var = row * size * size + col * size + num
        # For position (0,0) with number 1
        1: True,
        # For position (1,1) with number 2
        652: True,
        # Add some false assignments
        2: False,
        653: False,
        # Add invalid position to test bounds checking
        15626: True,  # Large number that would be out of grid bounds
    }

    # Call the solution reconstruction method
    solution = dpll_solver._solve_task()

    # Verify solution
    assert solution is not None
    assert isinstance(solution, list)
    assert len(solution) == 25
    assert all(len(row) == 25 for row in solution)

    # Check correct number placement
    assert solution[0][0] == 1  # First assignment
    assert solution[1][1] == 2  # Second assignment

    # Verify other positions are empty (0)
    assert solution[0][1] == 0
    assert solution[1][0] == 0


def test_solution_reconstruction_no_solution(dpll_solver):
    # Enable test mode
    dpll_solver._testing = True

    # Override dpll to return False
    dpll_solver.dpll = lambda: False

    # Verify no solution is returned
    assert dpll_solver._solve_task() is None
