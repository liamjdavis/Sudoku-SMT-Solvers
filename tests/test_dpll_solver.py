import signal
from unittest.mock import MagicMock, patch

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


@patch("signal.alarm")
def test_solve_valid(mock_alarm, dpll_solver):
    # Setup test puzzle
    dpll_solver.sudoku = [[1, 0], [0, 2]]
    dpll_solver.size = 2

    # Mock utils.get_var calls in solve()
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


@patch("signal.alarm")
def test_solve_timeout(mock_alarm, dpll_solver):
    with patch("signal.signal") as mock_signal:
        # Setup timeout
        mock_alarm.side_effect = TimeoutError("DPLL solver timed out")

        # Mock get_var to avoid early failure
        with patch("sudoku_smt_solvers.solvers.utils.get_var") as mock_get_var:
            mock_get_var.return_value = 1

            # Test timeout handling
            with pytest.raises(SudokuError) as exc_info:
                dpll_solver.solve()

            assert "timed out" in str(exc_info.value)
            mock_alarm.assert_called_once()


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


@patch.object(DPLLSolver, "dpll", side_effect=Exception("Mocked DPLL error"))
def test_solve_dpll_exception(mock_dpll, dpll_solver):
    with pytest.raises(SudokuError, match="Error solving puzzle: Mocked DPLL error"):
        dpll_solver.solve()
