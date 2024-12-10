import pytest
from sudoku_smt_solvers.solvers.utils import generate_cnf, get_var


def test_get_var_valid():
    # Test valid inputs
    assert get_var(0, 0, 1, 25) == 1
    assert get_var(0, 1, 1, 25) == 26
    assert get_var(1, 0, 1, 25) == 626
    assert get_var(24, 24, 25, 25) == 15625


@pytest.mark.parametrize(
    "row,col,num,size",
    [
        (-1, 0, 1, 25),
        (0, -1, 1, 25),
        (0, 0, 0, 25),
        (0, 0, 26, 25),
        (25, 0, 1, 25),
        (0, 25, 1, 25),
    ],
)
def test_get_var_invalid_inputs(row, col, num, size):
    with pytest.raises(ValueError):
        get_var(row, col, num, size)


def test_generate_cnf_valid_size():
    size = 25
    clauses = generate_cnf(size)

    # Test if clauses is not empty
    assert len(clauses) > 0

    # Test if all clauses are lists
    assert all(isinstance(clause, list) for clause in clauses)

    # Test if all variables are within valid range
    max_var = size * size * size
    assert all(all(abs(var) <= max_var for var in clause) for clause in clauses)


def test_generate_cnf_cell_constraints():
    size = 25
    clauses = generate_cnf(size)

    # Test if each cell has at least one number
    cell_clauses = [c for c in clauses if len(c) == size and all(x > 0 for x in c)]
    assert len(cell_clauses) == size * size

    # Test if each cell has at most one number
    at_most_one = [c for c in clauses if len(c) == 2 and all(x < 0 for x in c)]
    expected_at_most_one = size * size * (size * (size - 1) // 2)
    assert len(at_most_one) >= expected_at_most_one


def test_generate_cnf_row_constraints():
    size = 25
    clauses = generate_cnf(size)

    # Sample test for row constraints
    row = 0
    num = 1
    row_clauses = [
        c
        for c in clauses
        if len(c) == 2
        and c[0] == -get_var(row, 0, num, size)
        and c[1] == -get_var(row, 1, num, size)
    ]
    assert len(row_clauses) > 0


def test_generate_cnf_column_constraints():
    size = 25
    clauses = generate_cnf(size)

    # Sample test for column constraints
    col = 0
    num = 1
    col_clauses = [
        c
        for c in clauses
        if len(c) == 2
        and c[0] == -get_var(0, col, num, size)
        and c[1] == -get_var(1, col, num, size)
    ]
    assert len(col_clauses) > 0


def test_generate_cnf_block_constraints():
    size = 25
    block_size = 5
    clauses = generate_cnf(size)

    # Sample test for block constraints
    block_clauses = [
        c
        for c in clauses
        if len(c) == 2
        and c[0] == -get_var(0, 0, 1, size)
        and c[1] == -get_var(0, 1, 1, size)
    ]
    assert len(block_clauses) > 0


@pytest.mark.parametrize(
    "size",
    [
        0,
        -1,
        26,
        24,
    ],
)
def test_generate_cnf_invalid_size(size):
    with pytest.raises(ValueError):
        generate_cnf(size)


def test_generate_cnf_memory_limit():
    # Test with a large but valid size to check memory handling
    size = 25
    try:
        clauses = generate_cnf(size)
        assert len(clauses) > 0
    except MemoryError:
        pytest.skip("Not enough memory to run this test")


def test_clauses_consistency():
    size = 25
    clauses = generate_cnf(size)

    # All clauses should be non-empty
    assert all(len(clause) > 0 for clause in clauses)

    # All variables should be non-zero
    assert all(all(var != 0 for var in clause) for clause in clauses)

    # Variables should be within bounds
    max_var = size * size * size
    assert all(all(-max_var <= var <= max_var for var in clause) for clause in clauses)
