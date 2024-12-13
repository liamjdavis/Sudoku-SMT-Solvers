import os
import json
import pytest
from sudoku_smt_solvers.benchmarks.benchmark_runner import BenchmarkRunner
from unittest.mock import Mock, patch


@pytest.fixture
def mock_create_solver(mock_solver):
    with patch("sudoku_smt_solvers.benchmarks.benchmark_runner.create_solver") as mock:
        mock.return_value = mock_solver
        yield mock


@pytest.fixture
def mock_solver_response():
    return {"status": "sat", "solve_time": 0.1, "is_correct": True, "propagations": 100}


@pytest.fixture
def mock_solver(mock_solver_response):
    solver = Mock()
    solver.solve.return_value = mock_solver_response
    return solver


@pytest.fixture
def mock_solver_factory(mock_solver):
    with patch(
        "sudoku_smt_solvers.benchmarks.benchmark_runner.CVC5Solver"
    ) as mock_cvc5, patch(
        "sudoku_smt_solvers.benchmarks.benchmark_runner.DPLLSolver"
    ) as mock_dpll, patch(
        "sudoku_smt_solvers.benchmarks.benchmark_runner.DPLLTSolver"
    ) as mock_dpllt, patch(
        "sudoku_smt_solvers.benchmarks.benchmark_runner.Z3Solver"
    ) as mock_z3:
        mock_cvc5.return_value = mock_solver
        mock_dpll.return_value = mock_solver
        mock_dpllt.return_value = mock_solver
        mock_z3.return_value = mock_solver

        yield {
            "CVC5": mock_cvc5,
            "DPLL": mock_dpll,
            "DPLL(T)": mock_dpllt,
            "Z3": mock_z3,
        }


@pytest.fixture
def benchmark_runner(tmp_path):
    # Create temporary directories for tests
    puzzles_dir = tmp_path / "puzzles"
    solutions_dir = tmp_path / "solutions"
    results_dir = tmp_path / "results"

    puzzles_dir.mkdir()
    solutions_dir.mkdir()

    return BenchmarkRunner(
        puzzles_dir=str(puzzles_dir),
        solutions_dir=str(solutions_dir),
        results_dir=str(results_dir),
    )


@pytest.fixture
def sample_puzzle():
    # Create a 25x25 grid with some initial values
    grid = [[0] * 25 for _ in range(25)]
    # Add some sample values
    grid[0][0] = 1
    grid[0][1] = 2
    return grid


@pytest.fixture
def sample_solution():
    # Create a 25x25 grid solution
    grid = [[((i + j) % 25) + 1 for j in range(25)] for i in range(25)]
    return grid


def test_init(benchmark_runner):
    assert os.path.exists(benchmark_runner.results_dir)
    assert "CVC5" in benchmark_runner.solvers
    assert "DPLL" in benchmark_runner.solvers
    assert "DPLL(T)" in benchmark_runner.solvers
    assert "Z3" in benchmark_runner.solvers


def test_load_puzzle(benchmark_runner, sample_puzzle):
    # Create a test puzzle file
    puzzle_path = os.path.join(benchmark_runner.puzzles_dir, "test.json")
    with open(puzzle_path, "w") as f:
        json.dump({"puzzle": sample_puzzle}, f)

    loaded_puzzle = benchmark_runner.load_puzzle("test")
    assert loaded_puzzle == sample_puzzle


def test_load_puzzle_error(benchmark_runner):
    assert benchmark_runner.load_puzzle("nonexistent") is None


def test_load_solution(benchmark_runner, sample_solution):
    # Create a test solution file
    solution_path = os.path.join(benchmark_runner.solutions_dir, "test.json")
    with open(solution_path, "w") as f:
        json.dump({"solution": sample_solution}, f)

    loaded_solution = benchmark_runner.load_solution("test")
    assert loaded_solution == sample_solution


def test_load_solution_error(benchmark_runner):
    assert benchmark_runner.load_solution("nonexistent") is None


def test_validate_solution(benchmark_runner, sample_solution):
    assert benchmark_runner.validate_solution(sample_solution, sample_solution) is True
    assert benchmark_runner.validate_solution(None, sample_solution) is False
    assert benchmark_runner.validate_solution(sample_solution, None) is False


def test_run_solver(
    benchmark_runner, sample_puzzle, sample_solution, mock_solver, mock_solver_response
):
    # Set up the mock solver with required attributes
    mock_solver.solve.return_value = sample_solution  # Return the actual solution
    mock_solver.solve_time = 0.1
    mock_solver.propagated_clauses = 100

    # Patch the Z3Solver class in the benchmark_runner's solvers dict
    benchmark_runner.solvers["Z3"] = lambda x: mock_solver

    result = benchmark_runner.run_solver("Z3", sample_puzzle, sample_solution)

    assert result == {
        "status": "sat",
        "solve_time": 0.1,
        "is_correct": True,
        "propagations": 100,
    }
    mock_solver.solve.assert_called_once()


def test_run_solver_error(benchmark_runner, sample_puzzle, sample_solution):
    # Test with invalid solver name
    with pytest.raises(KeyError):
        benchmark_runner.run_solver("InvalidSolver", sample_puzzle, sample_solution)


def test_run_benchmarks(
    benchmark_runner, mock_solver_factory, sample_puzzle, sample_solution
):
    # Create test puzzle and solution files
    puzzle_path = os.path.join(benchmark_runner.puzzles_dir, "test.json")
    solution_path = os.path.join(benchmark_runner.solutions_dir, "test.json")

    with open(puzzle_path, "w") as f:
        json.dump({"puzzle": sample_puzzle}, f)
    with open(solution_path, "w") as f:
        json.dump({"solution": sample_solution}, f)

    benchmark_runner.run_benchmarks()

    # Check if results files were created
    results_files = os.listdir(benchmark_runner.results_dir)
    assert len([f for f in results_files if f.endswith(".json")]) == 1
    assert len([f for f in results_files if f.endswith(".csv")]) == 1

    # Verify JSON results structure
    json_file = next(f for f in results_files if f.endswith(".json"))
    with open(os.path.join(benchmark_runner.results_dir, json_file)) as f:
        results = json.load(f)

    for solver in benchmark_runner.solvers:
        assert solver in results
        assert "puzzles" in results[solver]
        assert "stats" in results[solver]

        stats = results[solver]["stats"]
        assert all(
            key in stats
            for key in [
                "total_puzzles",
                "solved_count",
                "correct_count",
                "total_time",
                "total_propagations",
                "avg_time",
                "avg_propagations",
            ]
        )
