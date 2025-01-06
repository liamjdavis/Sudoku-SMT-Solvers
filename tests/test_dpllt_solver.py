import pytest
import json
import os
from sudoku_smt_solvers.solvers.dpllt_solver import DPLLTSolver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_puzzle():
    with open("tests/data/valid_partial_grid_puzzle.json", "r") as f:
        return json.load(f)


@pytest.fixture
def valid_solution():
    with open("tests/data/valid_partial_grid_solution.json", "r") as f:
        return json.load(f)


def test_init_valid(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    assert solver.size == 25
    assert solver.sudoku == valid_puzzle


def test_init_invalid_grid():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        DPLLTSolver([[]])


def test_add_sudoku_clauses(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    solver.add_sudoku_clauses()
    assert len(solver.cnf.clauses) > 0
    assert solver.propagated_clauses > 0


def test_theory_propagation(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    conflict = solver.theory_propagation()
    assert conflict is None  # Valid puzzle should not have immediate conflicts


def test_extract_solution(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    # Create a mock model - just using first row as example
    model = [1, -2, 3]  # Simplified model for testing
    solution = solver.extract_solution(model)
    assert isinstance(solution, list)
    assert len(solution) == 25
    assert len(solution[0]) == 25


def test_validate_solution(valid_puzzle, valid_solution):
    solver = DPLLTSolver(valid_puzzle)
    assert solver.validate_solution(valid_solution) == True


def test_validate_invalid_solution(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    invalid_solution = [[1] * 25 for _ in range(25)]  # All 1s is invalid
    assert solver.validate_solution(invalid_solution) == False


def test_solve_complete(valid_puzzle, valid_solution):
    solver = DPLLTSolver(valid_puzzle)
    solution = solver.solve()
    assert solution is not None
    assert solver.validate_solution(solution)
    # Check if solution matches known valid solution
    assert solution == valid_solution


def test_solve_metrics(valid_puzzle):
    solver = DPLLTSolver(valid_puzzle)
    solution = solver.solve()
    assert solver.propagated_clauses > 0
