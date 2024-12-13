import pytest
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch
from sudoku_smt_solvers.benchmarks.sudoku_generator.sudoku_generator import (
    SudokuGenerator,
)


# Add these new fixtures
@pytest.fixture
def mock_hole_digger():
    with patch(
        "sudoku_smt_solvers.benchmarks.sudoku_generator.sudoku_generator.HoleDigger"
    ) as mock:
        mock_instance = Mock()
        mock_instance.dig_holes.return_value = [
            [1, 0],
            [3, 4],
        ]  # Sample puzzle with one hole
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_las_vegas():
    with patch(
        "sudoku_smt_solvers.benchmarks.sudoku_generator.sudoku_generator.LasVegasGenerator"
    ) as mock:
        yield mock


@pytest.fixture
def temp_dirs(tmp_path):
    puzzles_dir = tmp_path / "puzzles"
    solutions_dir = tmp_path / "solutions"
    return str(puzzles_dir), str(solutions_dir)


@pytest.fixture
def mock_datetime():
    with patch(
        "sudoku_smt_solvers.benchmarks.sudoku_generator.sudoku_generator.datetime"
    ) as mock:
        mock.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        yield mock


@pytest.fixture
def sample_grid():
    return [[1, 2], [3, 4]]  # Small 2x2 grid for testing


def test_generate_complete_workflow(
    mock_hole_digger, mock_las_vegas, temp_dirs, mock_datetime
):
    puzzles_dir, solutions_dir = temp_dirs

    # Setup test data
    solution = [[1, 2], [3, 4]]
    puzzle = [[1, 0], [3, 4]]

    # Configure mocks
    mock_las_vegas_instance = Mock()
    mock_las_vegas_instance.generate.return_value = solution
    mock_las_vegas.return_value = mock_las_vegas_instance

    # Create generator and test
    generator = SudokuGenerator(
        size=2, puzzles_dir=puzzles_dir, solutions_dir=solutions_dir
    )

    result_puzzle, result_solution, puzzle_id = generator.generate()

    # Verify results
    assert result_puzzle == puzzle
    assert result_solution == solution
    assert puzzle_id == "sudoku_2x2_Medium_20240101_120000"
