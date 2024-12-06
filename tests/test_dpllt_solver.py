import pytest
import signal
from unittest.mock import Mock, patch, create_autospec
from sudoku_smt_solvers.solvers.dpllt_solver import DPLLTSolver
from sudoku_smt_solvers.solvers.utils import generate_cnf, get_var
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_empty_grid():
    return [[0] * 25 for _ in range(25)]


@pytest.fixture
def valid_partial_grid():
    grid = [[0] * 25 for _ in range(25)]
    grid[0][0] = 1
    return grid


def test_constructor_valid_input(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    assert solver.size == 25
    assert solver.timeout == 120
    assert len(solver.clauses) == 0
    assert len(solver.assignments) == 0


def test_constructor_invalid_timeout(valid_empty_grid):  # Pass fixture as parameter
    with pytest.raises(SudokuError, match="Timeout must be positive"):
        DPLLTSolver(valid_empty_grid, timeout=0)


def test_constructor_invalid_grid_none():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        DPLLTSolver(None)


def test_constructor_invalid_grid_size():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        DPLLTSolver([[0] * 24 for _ in range(24)])


@patch("signal.signal")
@patch("signal.alarm")
def test_solve_timeout(mock_alarm, mock_signal, valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid, timeout=1)
    mock_alarm.side_effect = TimeoutError()

    with pytest.raises(SudokuError, match="Solver timed out"):
        solver.solve()


def test_analyze_conflict(valid_empty_grid):  # Pass fixture as parameter
    solver = DPLLTSolver(valid_empty_grid)
    solver.decision_level = 2
    solver.variable_level = {1: 2, 2: 1, 3: 0}

    learned, level = solver.analyze_conflict([1, -2, 3])
    assert level == 1
    assert -2 in learned


def test_propagate_theory_no_conflict(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    assert solver.propagate_theory() is None


def test_dpll_t_satisfiable(valid_partial_grid, monkeypatch):
    solver = DPLLTSolver(valid_partial_grid)

    # Mock find_unit_clause method
    def mock_find_unit():
        return None if not solver.clauses else solver.clauses[0][0]

    monkeypatch.setattr(solver, "find_unit_clause", mock_find_unit)

    solver.clauses = [[1]]
    solver.assignments = {}
    assert solver.dpll_t() is True


def test_dpll_t_unsatisfiable(valid_empty_grid, monkeypatch):
    solver = DPLLTSolver(valid_empty_grid)

    # Mock find_unit_clause method
    def mock_find_unit():
        return None if not solver.clauses else solver.clauses[0][0]

    monkeypatch.setattr(solver, "find_unit_clause", mock_find_unit)

    solver.clauses = [[1], [-1]]
    assert solver.dpll_t() is False


def test_backtrack(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True, 2: True, 3: True}
    solver.variable_level = {1: 0, 2: 1, 3: 2}
    solver.decision_level = 2

    solver.backtrack(1)
    assert solver.decision_level == 1
    assert len(solver.assignments) == 2
    assert 3 not in solver.assignments


def test_choose_variable(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1, 2], [-2, 3]]
    var = solver.choose_variable()
    assert isinstance(var, int)
    assert var > 0


def test_update_clauses(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1, 2], [-1, 3]]

    assert solver.update_clauses(1) is True
    assert len(solver.clauses) == 1
    assert solver.assignments[1] is True


def test_process_clause(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True}
    new_clauses = []

    assert solver.process_clause([1, 2], new_clauses) is True
    assert len(new_clauses) == 0


def test_extract_solution(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {get_var(0, 0, 1, 25): True}

    solution = solver.extract_solution()
    assert solution[0][0] == 1


def test_timeout_handler(valid_empty_grid):
    solver = DPLLTSolver(valid_empty_grid)
    with pytest.raises(TimeoutError):
        solver.timeout_handler(signal.SIGALRM, None)


def test_init_cnf_generation_error(valid_empty_grid, monkeypatch):
    """Test error handling in CNF generation (lines 27-28)"""

    def mock_generate_cnf(*args):
        raise Exception("CNF generation failed")

    monkeypatch.setattr(
        "sudoku_smt_solvers.solvers.dpllt_solver.generate_cnf", mock_generate_cnf
    )

    with pytest.raises(SudokuError, match="Error generating CNF"):
        DPLLTSolver(valid_empty_grid)


def test_analyze_conflict_empty_learned(valid_empty_grid):
    """Test analyze_conflict with no learned clauses (lines 56, 63)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.decision_level = 1
    solver.variable_level = {1: 0}  # All variables at level 0

    learned, level = solver.analyze_conflict([1])
    assert learned == []
    assert level == 0


def test_analyze_conflict_current_level(valid_empty_grid):
    """Test analyze_conflict with current level variables (lines 76-79)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.decision_level = 2
    solver.variable_level = {1: 2, 2: 2}  # Variables at current level

    learned, level = solver.analyze_conflict([1, 2])
    assert learned == []
    assert level == 0


def test_propagate_theory_with_assignments(valid_empty_grid):
    """Test theory propagation with assignments (lines 84-89)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {get_var(0, 0, 1, 25): True, get_var(0, 0, 2, 25): True}

    conflict = solver.propagate_theory()
    assert conflict is None


def test_dpll_t_unit_propagation_conflict(valid_empty_grid):
    """Test DPLL(T) unit propagation with conflict (lines 93-94)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1], [-1]]
    result = solver.dpll_t()
    assert result is False


def test_dpll_t_all_clauses_satisfied(valid_empty_grid):
    """Test DPLL(T) with all clauses satisfied (line 139)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = []
    solver.learned_clauses = []
    result = solver.dpll_t()
    assert result is True


def test_dpll_t_backtracking_false_branch(valid_empty_grid):
    """Test DPLL(T) backtracking false branch (lines 143-146)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1, -2], [-1, -2]]  # Only satisfiable with var 2 = False
    result = solver.dpll_t()
    assert result is True


def test_find_pure_literal_complex(valid_empty_grid):
    """Test find_pure_literal with complex clause set (lines 159-168)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1, 2], [1, 3], [1, 4]]  # 1 appears only positively
    solver.learned_clauses = [[2, -3], [2, 4]]
    pure = solver.find_pure_literal()
    assert pure == 1


def test_update_clauses_with_learned(valid_empty_grid):
    """Test update_clauses with learned clauses (lines 176, 178)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1, 2]]
    solver.learned_clauses = [[-1, 3]]
    result = solver.update_clauses(1)
    assert result is True
    assert len(solver.learned_clauses) == 1


def test_solve_exception_handling(valid_empty_grid, monkeypatch):
    """Test solve exception handling (lines 238-242)"""
    solver = DPLLTSolver(valid_empty_grid)

    def mock_dpll_t():
        raise Exception("Unexpected error")

    monkeypatch.setattr(solver, "dpll_t", mock_dpll_t)

    with pytest.raises(SudokuError, match="Error solving puzzle"):
        solver.solve()


def test_init_cnf_generation_error(valid_empty_grid, monkeypatch):
    """Test error handling in CNF generation"""

    def mock_generate_cnf(*args):
        raise Exception("CNF generation failed")

    monkeypatch.setattr(
        "sudoku_smt_solvers.solvers.dpllt_solver.generate_cnf", mock_generate_cnf
    )

    with pytest.raises(SudokuError, match="Error generating CNF"):
        DPLLTSolver(valid_empty_grid)


def test_analyze_conflict_backtrack_level(valid_empty_grid):
    """Test backtrack level calculation in analyze_conflict"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.decision_level = 3
    solver.variable_level = {1: 1, 2: 2, 3: 3}

    learned, level = solver.analyze_conflict([1, 2, 3])
    assert learned == [1, 2]  # Variables from lower levels
    assert level == 2  # Max level of learned variables


def test_dpll_t_assignment_backup(valid_empty_grid):
    """Test assignment backup during DPLL(T) branching"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True, 2: False}
    solver.clauses = [[3, 4]]

    original_assignments = solver.assignments.copy()
    solver.dpll_t()  # This should trigger assignment backup

    # Verify assignments were properly backed up during branching
    assert solver.assignments == {1: True, 2: False, 3: True}


def test_backtrack_variable_filtering(valid_empty_grid):
    """Test selective variable filtering during backtrack"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True, 2: True, 3: True}
    solver.variable_level = {1: 0, 2: 1, 3: 2}

    solver.backtrack(1)  # Backtrack to level 1

    # Variables at level ≤ 1 should remain
    assert 1 in solver.assignments
    assert 2 in solver.assignments
    assert 3 not in solver.assignments


def test_find_pure_literal_with_assignments(valid_empty_grid):
    """Test pure literal finding with assigned variables"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True}  # Variable 1 is already assigned
    solver.clauses = [[1, 2], [2, 3], [-3]]  # 2 is pure

    pure = solver.find_pure_literal()
    assert pure == 2  # Should find 2 as pure, ignoring assigned variable 1


def test_process_learned_clauses(valid_empty_grid):
    """Test processing of learned clauses"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True}
    solver.learned_clauses = [[-1, 2], [1, 3]]
    new_learned = []

    # Process learned clauses
    result = solver.process_clause([-1, 2], new_learned)
    assert result is True
    assert len(new_learned) == 1
    assert new_learned[0] == [2]


def test_analyze_conflict_multiple_levels(valid_empty_grid):
    """Test analyze_conflict with variables at different levels (lines 76-79)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.decision_level = 3
    solver.variable_level = {1: 1, 2: 3, 3: 2}

    learned, level = solver.analyze_conflict([1, 2, 3])
    assert level == 2  # Should be max level of learned variables
    assert 1 in [abs(x) for x in learned]  # Should include level 1 variable
    assert 3 in [abs(x) for x in learned]  # Should include level 2 variable


def test_dpll_t_unit_propagation_fail(valid_empty_grid):
    """Test DPLL(T) with failing unit propagation (line 94)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = [[1], [-1]]  # Contradictory unit clauses

    result = solver.dpll_t()
    assert result is False


def test_dpll_t_satisfied_clauses(valid_empty_grid):
    """Test DPLL(T) when all clauses are satisfied (line 139)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.clauses = []
    solver.learned_clauses = []
    solver.assignments = {1: True}

    result = solver.dpll_t()
    assert result is True


def test_process_clause_with_learned(valid_empty_grid):
    """Test process_clause with learned clauses (line 200)"""
    solver = DPLLTSolver(valid_empty_grid)
    solver.assignments = {1: True, 2: False}
    learned_clause = [-1, 2, 3]
    new_learned = []

    result = solver.process_clause(learned_clause, new_learned)
    assert result is True
    assert len(new_learned) == 1
    assert new_learned[0] == [3]  # Only unassigned variable remains


def test_solve_unexpected_error(valid_empty_grid, monkeypatch):
    """Test solve method error handling (line 240)"""
    solver = DPLLTSolver(valid_empty_grid)

    def mock_dpll_t():
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(solver, "dpll_t", mock_dpll_t)

    with pytest.raises(SudokuError, match="Error solving puzzle"):
        solver.solve()


def test_solve_timeout_error(valid_empty_grid):
    """Test solve method timeout handling (line 248)"""
    solver = DPLLTSolver(valid_empty_grid, timeout=1)

    def slow_dpll_t():
        import time

        time.sleep(2)  # Force timeout
        return True

    solver.dpll_t = slow_dpll_t

    with pytest.raises(SudokuError, match="Solver timed out"):
        solver.solve()


def test_solve_returns_none_on_initial_conflict(valid_partial_grid, monkeypatch):
    """Test solve() returns None when initial assignments cause conflict"""
    solver = DPLLTSolver(valid_partial_grid)

    # Mock update_clauses to return False to simulate initial conflict
    def mock_update_clauses(assignment):
        return False

    monkeypatch.setattr(solver, "update_clauses", mock_update_clauses)

    # solve() should return None when initial assignments conflict
    assert solver.solve() is None


def test_update_clauses_returns_false(valid_empty_grid, monkeypatch):
    """Test update_clauses returns False when process_clause fails"""
    solver = DPLLTSolver(valid_empty_grid)

    # Set up test conditions
    solver.clauses = [[1, 2], [3, 4]]  # Add some clauses

    # Mock process_clause to return False
    def mock_process_clause(clause, new_clauses):
        return False

    monkeypatch.setattr(solver, "process_clause", mock_process_clause)

    # Call update_clauses with a test assignment
    result = solver.update_clauses(1)

    # Verify return value is False
    assert result is False

    # Verify assignments were still made before returning False
    assert solver.assignments[1] is True
    assert solver.variable_level[1] == solver.decision_level
