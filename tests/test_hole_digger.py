import pytest
from sudoku_smt_solvers.benchmarks.sudoku_generator.hole_digger import (
    HoleDigger,
    difficulty_givensRange_mapping,
)
import random

# Mock mappings for 4x4 grid
mock_difficulty_givensRange_mapping = {
    "Extremely Easy": [12, 16],
    "Easy": [10, 12],
    "Medium": [8, 10],
    "Difficult": [6, 8],
    "Evil": [4, 6],
}

mock_difficulty_lower_bound_mapping = {
    "Extremely Easy": 3,
    "Easy": 2,
    "Medium": 2,
    "Difficult": 1,
    "Evil": 0,
}


@pytest.fixture(autouse=True)
def mock_mappings(monkeypatch):
    monkeypatch.setattr(
        "sudoku_smt_solvers.benchmarks.sudoku_generator.hole_digger.difficulty_givensRange_mapping",
        mock_difficulty_givensRange_mapping,
    )
    monkeypatch.setattr(
        "sudoku_smt_solvers.benchmarks.sudoku_generator.hole_digger.difficulty_lower_bound_mapping",
        mock_difficulty_lower_bound_mapping,
    )


@pytest.fixture
def hole_digger_easy():
    # 4x4 puzzle setup
    puzzle = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
    # For Easy difficulty:
    # - num_givens should be between 274-381 scaled for 4x4
    # - lower_bound is 11 scaled for 4x4
    scaled_min = max(int((274 / 729) * 16), 8)  # Scale from 27x27 to 4x4
    scaled_max = max(int((381 / 729) * 16), 12)

    digger = HoleDigger(puzzle, "Easy")
    digger.remaining_cells = 16  # 4x4 grid full
    digger.num_givens = scaled_min  # Minimum required givens
    return digger


def test_initialization(hole_digger_easy):
    assert hole_digger_easy.size == 4
    assert hole_digger_easy.difficulty == "Easy"
    assert hole_digger_easy.pattern == "random"
    assert hole_digger_easy.can_be_dug == {(i, j) for i in range(4) for j in range(4)}
    assert hole_digger_easy.remaining_cells == 16


def test_random_pattern(hole_digger_easy):
    random.seed(42)  # For reproducibility
    sequence = hole_digger_easy._random_pattern()
    assert len(sequence) == 16
    assert len(set(sequence)) == 16  # All positions are unique
    assert all(0 <= x < 4 and 0 <= y < 4 for x, y in sequence)


def test_jumping_pattern(hole_digger_easy):
    sequence = hole_digger_easy._jumping_pattern()
    assert len(sequence) == 8  # For 4x4, jumping pattern covers half cells
    assert len(set(sequence)) == 8  # All positions are unique


def test_s_pattern(hole_digger_easy):
    sequence = hole_digger_easy._s_pattern()
    assert len(sequence) == 16
    assert sequence[0] == (0, 0)  # Starts at top-left
    assert sequence[-1] == (3, 0)  # Ends at bottom-left for 4x4


def test_linear_pattern(hole_digger_easy):
    sequence = hole_digger_easy._linear_pattern()
    assert len(sequence) == 16
    assert sequence[0] == (0, 0)
    assert sequence[-1] == (3, 3)


def test_pass_restrictions(hole_digger_easy):
    # Test initial state
    assert hole_digger_easy.pass_restrictions(0, 0) == False

    # Test remaining cells restriction
    hole_digger_easy.remaining_cells = hole_digger_easy.num_givens
    assert hole_digger_easy.pass_restrictions(0, 0) == False

    # Reset and test row constraint
    hole_digger_easy.remaining_cells = 16


def test_has_unique_solution(hole_digger_easy):
    assert hole_digger_easy.has_unique_solution() == True
    # Create non-unique puzzle by removing too many numbers
    hole_digger_easy.puzzle = [[0, 0, 0, 4], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 1]]
    assert hole_digger_easy.has_unique_solution() == False


def test_dig_holes_process(hole_digger_easy):
    original_puzzle = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]

    for difficulty in ["Extremely Easy", "Easy", "Medium", "Difficult", "Evil"]:
        digger = HoleDigger(original_puzzle, difficulty)
        result = digger.dig_holes()

        # Check resulting puzzle has correct number of givens
        givens = sum(cell != 0 for row in result for cell in row)
        min_givens, max_givens = mock_difficulty_givensRange_mapping[difficulty]

        if max_givens != float("inf"):
            assert min_givens <= givens <= max_givens
        else:
            assert givens >= min_givens


def test_difficulty_mappings():
    assert "Extremely Easy" in difficulty_givensRange_mapping
    assert "Evil" in difficulty_givensRange_mapping
    assert difficulty_givensRange_mapping["Easy"][0] == 274
    assert difficulty_givensRange_mapping["Evil"][1] == 211
