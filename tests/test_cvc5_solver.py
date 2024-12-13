import pytest
from sudoku_smt_solvers.solvers.cvc5_solver import CVC5Solver
from sudoku_smt_solvers.solvers.sudoku_error import SudokuError


@pytest.fixture
def valid_empty_grid():
    return [[0 for _ in range(25)] for _ in range(25)]


@pytest.fixture
def valid_partial_grid():
    grid = [
        [
            15,
            19,
            20,
            1,
            3,
            7,
            10,
            13,
            21,
            25,
            6,
            8,
            17,
            4,
            18,
            16,
            23,
            24,
            9,
            22,
            14,
            11,
            2,
            12,
            5,
        ],
        [
            4,
            25,
            16,
            8,
            5,
            20,
            6,
            1,
            2,
            22,
            19,
            11,
            12,
            23,
            15,
            21,
            14,
            13,
            18,
            10,
            24,
            9,
            7,
            17,
            3,
        ],
        [
            12,
            2,
            13,
            9,
            17,
            23,
            16,
            4,
            24,
            5,
            20,
            1,
            3,
            21,
            14,
            15,
            8,
            19,
            11,
            7,
            22,
            25,
            10,
            18,
            6,
        ],
        [
            7,
            21,
            23,
            22,
            6,
            9,
            14,
            3,
            11,
            18,
            5,
            16,
            24,
            25,
            10,
            4,
            1,
            17,
            12,
            2,
            15,
            20,
            13,
            19,
            8,
        ],
        [
            18,
            11,
            10,
            24,
            14,
            8,
            15,
            12,
            17,
            19,
            7,
            2,
            9,
            13,
            22,
            20,
            3,
            6,
            25,
            5,
            1,
            23,
            4,
            21,
            16,
        ],
        [
            11,
            17,
            18,
            6,
            15,
            25,
            4,
            20,
            22,
            23,
            8,
            3,
            10,
            1,
            16,
            13,
            2,
            5,
            19,
            21,
            7,
            12,
            14,
            9,
            24,
        ],
        [
            19,
            5,
            22,
            23,
            24,
            3,
            7,
            16,
            8,
            10,
            15,
            13,
            14,
            18,
            2,
            12,
            25,
            11,
            20,
            9,
            4,
            6,
            17,
            1,
            21,
        ],
        [
            3,
            1,
            12,
            21,
            9,
            18,
            19,
            2,
            13,
            6,
            25,
            7,
            11,
            17,
            20,
            10,
            15,
            4,
            24,
            14,
            5,
            16,
            8,
            22,
            23,
        ],
        [
            2,
            10,
            14,
            13,
            16,
            11,
            24,
            9,
            5,
            17,
            22,
            23,
            21,
            12,
            4,
            3,
            7,
            1,
            8,
            6,
            18,
            15,
            19,
            20,
            25,
        ],
        [
            20,
            8,
            7,
            25,
            4,
            12,
            21,
            14,
            1,
            15,
            24,
            5,
            19,
            6,
            9,
            18,
            22,
            16,
            23,
            17,
            2,
            10,
            3,
            13,
            11,
        ],
        [
            22,
            23,
            1,
            16,
            19,
            14,
            8,
            6,
            20,
            11,
            18,
            17,
            4,
            10,
            13,
            7,
            24,
            9,
            5,
            15,
            12,
            3,
            21,
            25,
            2,
        ],
        [
            14,
            15,
            21,
            18,
            12,
            16,
            2,
            5,
            9,
            7,
            23,
            25,
            20,
            19,
            1,
            17,
            13,
            10,
            22,
            3,
            6,
            24,
            11,
            8,
            4,
        ],
        [
            24,
            3,
            2,
            5,
            11,
            19,
            17,
            21,
            25,
            4,
            14,
            22,
            7,
            15,
            8,
            23,
            6,
            12,
            16,
            1,
            13,
            18,
            20,
            10,
            9,
        ],
        [
            9,
            13,
            4,
            17,
            10,
            1,
            18,
            22,
            23,
            3,
            12,
            6,
            2,
            24,
            5,
            8,
            11,
            20,
            21,
            25,
            16,
            19,
            15,
            14,
            7,
        ],
        [
            8,
            20,
            6,
            7,
            25,
            10,
            13,
            15,
            12,
            24,
            21,
            9,
            16,
            3,
            11,
            2,
            18,
            14,
            4,
            19,
            17,
            5,
            22,
            23,
            1,
        ],
        [
            16,
            14,
            19,
            2,
            20,
            15,
            3,
            11,
            10,
            9,
            13,
            21,
            1,
            22,
            17,
            6,
            12,
            23,
            7,
            24,
            8,
            4,
            25,
            5,
            18,
        ],
        [
            23,
            7,
            11,
            4,
            13,
            24,
            12,
            25,
            6,
            16,
            10,
            14,
            15,
            8,
            19,
            5,
            20,
            21,
            17,
            18,
            3,
            1,
            9,
            2,
            22,
        ],
        [
            6,
            24,
            25,
            10,
            8,
            22,
            23,
            18,
            7,
            14,
            2,
            4,
            5,
            20,
            3,
            9,
            19,
            15,
            1,
            13,
            11,
            21,
            12,
            16,
            17,
        ],
        [
            5,
            22,
            9,
            15,
            1,
            2,
            20,
            17,
            4,
            21,
            16,
            12,
            18,
            7,
            23,
            11,
            10,
            25,
            3,
            8,
            19,
            14,
            6,
            24,
            13,
        ],
        [
            21,
            12,
            17,
            3,
            18,
            5,
            1,
            8,
            19,
            13,
            11,
            24,
            6,
            9,
            25,
            22,
            16,
            2,
            14,
            4,
            10,
            7,
            23,
            15,
            20,
        ],
        [
            10,
            4,
            3,
            12,
            22,
            21,
            25,
            24,
            18,
            8,
            1,
            20,
            13,
            5,
            6,
            14,
            17,
            7,
            15,
            23,
            9,
            2,
            16,
            11,
            19,
        ],
        [
            13,
            18,
            24,
            20,
            7,
            17,
            11,
            19,
            14,
            2,
            9,
            15,
            22,
            16,
            21,
            25,
            5,
            3,
            6,
            12,
            23,
            8,
            1,
            4,
            10,
        ],
        [
            1,
            6,
            8,
            14,
            2,
            13,
            9,
            7,
            15,
            20,
            4,
            18,
            23,
            11,
            24,
            19,
            21,
            22,
            10,
            16,
            25,
            17,
            5,
            3,
            12,
        ],
        [
            25,
            16,
            15,
            19,
            21,
            4,
            5,
            23,
            3,
            12,
            17,
            10,
            8,
            2,
            7,
            1,
            9,
            18,
            13,
            11,
            20,
            22,
            24,
            6,
            14,
        ],
        [
            17,
            9,
            5,
            11,
            23,
            6,
            22,
            10,
            16,
            1,
            3,
            19,
            25,
            14,
            12,
            24,
            4,
            8,
            2,
            20,
            21,
            13,
            18,
            7,
            15,
        ],
    ]
    grid[24][24] = 0
    grid[24][23] = 0
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
    solver.encode_puzzle()

    # Check satisfiability first
    assert solver.solver.checkSat().isSat()

    # Verify that the constraints for filled cells were added correctly
    model = solver.solver.getValue(solver.variables[0][0])
    assert model.getIntegerValue() == 1

    model = solver.solver.getValue(solver.variables[24][24])
    assert model.getIntegerValue() == 25


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


def test_validate_solution_invalid_dimensions():
    solver = CVC5Solver([[0] * 25 for _ in range(25)])
    invalid_solution = [[1] * 24 for _ in range(25)]
    assert not solver.validate_solution(invalid_solution)


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
