import pytest
import json
import os
from sudoku_smt_solvers.solvers.dpll_solver import DPLLSolver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_puzzle():
    with open("tests/data/valid_partial_grid_puzzle.json", "r") as f:
        return json.load(f)


@pytest.fixture
def valid_solution():
    with open("tests/data/valid_partial_grid_solution.json", "r") as f:
        return json.load(f)


def test_init_valid():
    """Test initialization with valid puzzle"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle)
    assert solver.size == 25
    assert solver.timeout == 120  # default timeout
    assert solver.propagated_clauses == 0


def test_init_invalid_timeout():
    """Test initialization with invalid timeout"""
    puzzle = [[0] * 25 for _ in range(25)]
    with pytest.raises(SudokuError, match="Timeout must be positive"):
        DPLLSolver(puzzle, timeout=0)


def test_init_invalid_puzzle():
    """Test initialization with invalid puzzle"""
    puzzle = [[0] * 24 for _ in range(24)]  # Wrong size
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        DPLLSolver(puzzle)


def test_clause_counting():
    """Test clause counter increment"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle)
    initial_count = solver.propagated_clauses
    solver._count_clause()
    assert solver.propagated_clauses == initial_count + 1


def test_add_sudoku_clauses():
    """Test clause generation"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle)
    solver.add_sudoku_clauses()
    assert solver.propagated_clauses > 0
    assert len(solver.cnf.clauses) > 0


def test_extract_solution():
    """Test solution extraction from SAT model"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle)
    # Create a simple model where cell (0,0) = 1
    model = [1] + [0] * (25 * 25 * 25)
    solution = solver.extract_solution(model)
    assert solution[0][0] == 1


def test_validate_solution_invalid_row():
    """Test solution validation with invalid row"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle)
    invalid_solution = [[1] * 25 for _ in range(25)]  # All 1s, invalid
    assert solver.validate_solution(invalid_solution) == False


def test_solve_valid_puzzle(valid_puzzle, valid_solution):
    """Test complete solve workflow with valid puzzle"""
    solver = DPLLSolver(valid_puzzle)
    solution = solver.solve()
    assert solution is not None
    assert solver.validate_solution(solution)
    # Verify solution matches known solution
    assert solution == valid_solution


def test_solve_timeout():
    """Test solver timeout"""
    puzzle = [[0] * 25 for _ in range(25)]
    solver = DPLLSolver(puzzle, timeout=1)
    with pytest.raises(SudokuError, match="Solving timed out"):
        solver.solve()
