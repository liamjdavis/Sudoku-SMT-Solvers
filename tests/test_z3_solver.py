import pytest
import json
import os
from solvers.z3_solver import Z3Solver
from solvers.utils.sudoku_error import SudokuError


@pytest.fixture
def valid_puzzle():
    with open("tests/data/valid_partial_grid_puzzle.json", "r") as f:
        return json.load(f)


@pytest.fixture
def valid_solution():
    with open("tests/data/valid_partial_grid_solution.json", "r") as f:
        return json.load(f)


def test_solver_init_valid(valid_puzzle):
    solver = Z3Solver(valid_puzzle)
    assert solver.size == 25
    assert solver.propagated_clauses == 0


def test_solver_init_invalid_grid():
    with pytest.raises(SudokuError, match="Invalid Sudoku puzzle"):
        Z3Solver([[]])


def test_create_variables(valid_puzzle):
    solver = Z3Solver(valid_puzzle)
    solver.create_variables()
    assert len(solver.variables) == 25
    assert len(solver.variables[0]) == 25
    assert str(solver.variables[0][0]) == "x_0_0"


def test_validate_solution_valid(valid_puzzle, valid_solution):
    solver = Z3Solver(valid_puzzle)
    assert solver.validate_solution(valid_solution) == True


def test_validate_solution_invalid_range(valid_puzzle):
    invalid_solution = [[0 for _ in range(25)] for _ in range(25)]
    solver = Z3Solver(valid_puzzle)
    assert solver.validate_solution(invalid_solution) == False


def test_validate_solution_invalid_row(valid_puzzle, valid_solution):
    invalid_solution = [row[:] for row in valid_solution]
    invalid_solution[0][0] = invalid_solution[0][1]  # Duplicate in row
    solver = Z3Solver(valid_puzzle)
    assert solver.validate_solution(invalid_solution) == False


def test_validate_solution_invalid_column(valid_puzzle, valid_solution):
    invalid_solution = [row[:] for row in valid_solution]
    invalid_solution[1][0] = invalid_solution[0][0]  # Duplicate in column
    solver = Z3Solver(valid_puzzle)
    assert solver.validate_solution(invalid_solution) == False


def test_validate_solution_invalid_box(valid_puzzle, valid_solution):
    invalid_solution = [row[:] for row in valid_solution]
    invalid_solution[1][1] = invalid_solution[0][0]  # Duplicate in box
    solver = Z3Solver(valid_puzzle)
    assert solver.validate_solution(invalid_solution) == False


def test_solve_valid_puzzle(valid_puzzle, valid_solution):
    solver = Z3Solver(valid_puzzle)
    result = solver.solve()
    assert result is not None
    assert result == valid_solution
    assert solver.propagated_clauses > 0


def test_solve_unsolvable():
    # Create an unsolvable puzzle by putting same number twice in a row
    unsolvable_puzzle = [[0 for _ in range(25)] for _ in range(25)]
    unsolvable_puzzle[0][0] = 1
    unsolvable_puzzle[0][1] = 1
    solver = Z3Solver(unsolvable_puzzle)
    result = solver.solve()
    assert result is None


def test_propagated_clauses_counting(valid_puzzle):
    solver = Z3Solver(valid_puzzle)
    initial_clauses = solver.propagated_clauses
    solver.solve()
    assert solver.propagated_clauses > initial_clauses
