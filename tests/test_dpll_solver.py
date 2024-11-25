import pytest
from sudoku_smt_solvers.solvers.dpll_solver import DPLLSolver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError
import signal
from unittest.mock import patch, MagicMock


@pytest.fixture
def valid_25x25_puzzle():
    # Create a 25x25 puzzle with some initial values
    puzzle = [[0] * 25 for _ in range(25)]
    puzzle[0][0] = 1  # Add some known values
    puzzle[1][1] = 2
    return puzzle


def test_generate_cnf_exception():
    # Create a valid 25x25 Sudoku puzzle
    valid_puzzle = [[0] * 25 for _ in range(25)]

    # Mock the generate_cnf method to raise an exception
    with patch.object(
        DPLLSolver, "generate_cnf", side_effect=Exception("Mocked CNF generation error")
    ):
        with pytest.raises(
            SudokuError, match="Error generating CNF: Mocked CNF generation error"
        ):
            DPLLSolver(valid_puzzle)


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


def test_get_var(dpll_solver):
    var = dpll_solver.get_var(1, 2, 3)
    assert isinstance(var, int)
    # Test specific encoding
    assert var == 1 * 25 * 25 + 2 * 25 + 3


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
    # Create a minimal valid 2x2 puzzle with proper CNF constraints
    dpll_solver.sudoku = [[1, 0], [0, 2]]
    dpll_solver.size = 2
    dpll_solver.clauses = []
    dpll_solver.assignments = {}

    # Add minimal constraints for 2x2 puzzle
    dpll_solver.clauses = [
        [dpll_solver.get_var(0, 0, 1)],
        [dpll_solver.get_var(1, 1, 2)],
        [dpll_solver.get_var(0, 1, 1), dpll_solver.get_var(0, 1, 2)],
        [dpll_solver.get_var(1, 0, 1), dpll_solver.get_var(1, 0, 2)],
    ]

    solution = dpll_solver.solve()
    assert solution is not None
    assert solution[0][0] == 1
    assert solution[1][1] == 2


@patch("signal.alarm")
def test_solve_timeout(mock_alarm, dpll_solver):
    def mock_timeout_handler(*args):
        raise TimeoutError("Solver timed out")

    with patch("signal.signal") as mock_signal:
        dpll_solver.timeout_handler = mock_timeout_handler
        mock_alarm.side_effect = lambda x: mock_timeout_handler(None, None)

        with pytest.raises(SudokuError, match=".*timed out.*"):
            dpll_solver.solve()


def test_solve_invalid_puzzle():
    invalid_puzzle = "not a valid puzzle"
    with pytest.raises(SudokuError):
        solver = DPLLSolver(invalid_puzzle)
        solver.solve()


def test_generate_cnf_clauses(dpll_solver):
    # Test that CNF generation creates expected clause types
    assert len(dpll_solver.clauses) > 0
    # Test at least one clause of each type exists
    found_cell = False
    found_row = False
    found_col = False
    found_block = False

    for clause in dpll_solver.clauses:
        if len(clause) == 25:  # Cell constraint
            found_cell = True
        elif len(clause) == 2:  # Row/Col/Block constraints
            var1, var2 = abs(clause[0]), abs(clause[1])
            if var1 // (25 * 25) == var2 // (25 * 25):  # Same row
                found_row = True
            elif (var1 % (25 * 25)) // 25 == (var2 % (25 * 25)) // 25:  # Same column
                found_col = True
            else:
                found_block = True

    assert all([found_cell, found_row, found_col, found_block])


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
