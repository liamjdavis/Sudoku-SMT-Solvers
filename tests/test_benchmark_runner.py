import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open

from sudoku_smt_solvers.benchmarks.benchmark_runner import BenchmarkRunner


@pytest.fixture
def valid_puzzle():
    puzzle_path = "tests/data/valid_partial_grid_puzzle.json"
    with open(puzzle_path) as f:
        return json.load(f)


@pytest.fixture
def valid_solution():
    solution_path = "tests/data/valid_partial_grid_solution.json"
    with open(solution_path) as f:
        return json.load(f)


@pytest.fixture
def benchmark_runner(tmp_path):
    puzzles_dir = tmp_path / "puzzles"
    results_dir = tmp_path / "results"
    puzzles_dir.mkdir()

    return BenchmarkRunner(
        puzzles_dir=str(puzzles_dir), results_dir=str(results_dir), timeout=30
    )


def test_load_puzzle(benchmark_runner, valid_puzzle):
    puzzle_path = os.path.join(benchmark_runner.puzzles_dir, "test.json")
    with open(puzzle_path, "w") as f:
        json.dump({"puzzle": valid_puzzle}, f)

    loaded_puzzle = benchmark_runner.load_puzzle("test")
    assert loaded_puzzle == valid_puzzle


def test_load_puzzle_alternate_keys(benchmark_runner, valid_puzzle):
    puzzle_path = os.path.join(benchmark_runner.puzzles_dir, "test.json")

    # Test grid key
    with open(puzzle_path, "w") as f:
        json.dump({"grid": valid_puzzle}, f)
    assert benchmark_runner.load_puzzle("test") == valid_puzzle

    # Test gridc key
    with open(puzzle_path, "w") as f:
        json.dump({"gridc": valid_puzzle}, f)
    assert benchmark_runner.load_puzzle("test") == valid_puzzle


@patch("multiprocessing.get_context")
def test_run_solver_success(mock_get_context, benchmark_runner, valid_puzzle):
    # Mock multiprocessing
    mock_process = MagicMock()
    mock_queue = MagicMock()
    mock_queue.get_nowait.return_value = (True, 100)  # (result, propagations)

    mock_context = MagicMock()
    mock_context.Process.return_value = mock_process
    mock_context.Queue.return_value = mock_queue

    # Configure mock_process behavior
    mock_process.is_alive.return_value = False

    mock_get_context.return_value = mock_context

    result = benchmark_runner.run_solver("Z3", valid_puzzle)

    # Verify process interactions
    mock_process.start.assert_called_once()
    mock_process.join.assert_called_once_with(
        timeout=30
    )  # matches benchmark_runner timeout
    mock_process.is_alive.assert_called_once()

    assert result["status"] == "sat"
    assert "solve_time" in result
    assert result["propagations"] == 100


def test_run_benchmarks(benchmark_runner, valid_puzzle):
    # Create test puzzle file
    puzzle_path = os.path.join(benchmark_runner.puzzles_dir, "test.json")
    with open(puzzle_path, "w") as f:
        json.dump({"puzzle": valid_puzzle}, f)

    with patch.object(benchmark_runner, "run_solver") as mock_run_solver:
        mock_run_solver.return_value = {
            "status": "sat",
            "solve_time": 0.1,
            "propagations": 100,
        }

        benchmark_runner.run_benchmarks()

        results_files = os.listdir(benchmark_runner.results_dir)
        csv_files = [f for f in results_files if f.endswith(".csv")]

        assert len(csv_files) == 1

        # Verify CSV format
        csv_path = os.path.join(benchmark_runner.results_dir, csv_files[0])
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            assert header == [
                "solver",
                "puzzle_id",
                "status",
                "solve_time",
                "propagations",
            ]


def test_load_puzzle_no_valid_grid():
    # Setup
    puzzle_data = {"invalid_key": [[1, 2, 3]]}
    mock_json = json.dumps(puzzle_data)

    with (
        patch("builtins.open", mock_open(read_data=mock_json)),
        patch("builtins.print") as mock_print,
    ):
        runner = BenchmarkRunner()
        result = runner.load_puzzle("test_puzzle")

        # Assert
        assert result is None
        mock_print.assert_called_once_with(
            "No valid grid found in test_puzzle. Available keys: ['invalid_key']"
        )


def test_load_puzzle_file_error():
    # Setup
    with (
        patch("builtins.open", side_effect=Exception("File not found")),
        patch("builtins.print") as mock_print,
    ):
        runner = BenchmarkRunner()
        result = runner.load_puzzle("nonexistent_puzzle")

        # Assert
        assert result is None
        mock_print.assert_called_once_with(
            "Error loading puzzle nonexistent_puzzle: File not found"
        )
