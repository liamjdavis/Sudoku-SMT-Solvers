import json
import os
import time
from typing import Dict, List, Optional

from ..solvers import CVC5Solver, DPLLSolver, DPLLTSolver, Z3Solver


class BenchmarkRunner:
    def __init__(
        self,
        puzzles_dir: str = "benchmarks/puzzles",
        solutions_dir: str = "benchmarks/solutions",
        results_dir: str = "benchmarks/results",
    ):
        self.puzzles_dir = puzzles_dir
        self.solutions_dir = solutions_dir
        self.results_dir = results_dir
        self.solvers = {
            "CVC5": CVC5Solver,
            "DPLL": DPLLSolver,
            "DPLL(T)": DPLLTSolver,
            "Z3": Z3Solver,
        }
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

    def load_puzzle(self, puzzle_id: str) -> Optional[List[List[int]]]:
        """Load a puzzle from the puzzles directory."""
        puzzle_path = os.path.join(self.puzzles_dir, f"{puzzle_id}.json")
        try:
            with open(puzzle_path, "r") as f:
                data = json.load(f)
                return data["puzzle"]
        except Exception as e:
            print(f"Error loading puzzle {puzzle_id}: {e}")
            return None

    def load_solution(self, puzzle_id: str) -> Optional[List[List[int]]]:
        """Load a solution from the solutions directory."""
        solution_path = os.path.join(self.solutions_dir, f"{puzzle_id}.json")
        try:
            with open(solution_path, "r") as f:
                data = json.load(f)
                return data["solution"]
        except Exception as e:
            print(f"Error loading solution {puzzle_id}: {e}")
            return None

    def validate_solution(
        self, solution: List[List[int]], expected: List[List[int]]
    ) -> bool:
        """Validate if the solution matches the expected solution."""
        if not solution or not expected:
            return False
        return all(
            solution[i][j] == expected[i][j]
            for i in range(len(solution))
            for j in range(len(solution[0]))
        )

    def run_solver(
        self, solver_name: str, puzzle: List[List[int]], expected: List[List[int]]
    ) -> Dict:
        """Run a single solver on a puzzle and collect results."""
        solver_class = self.solvers[solver_name]
        solver = solver_class(puzzle)

        try:
            solution = solver.solve()
            solve_time = getattr(solver, "solve_time", getattr(solver, "timeout", 120))
            is_sat = solution is not None
            is_correct = self.validate_solution(solution, expected) if is_sat else False
            propagations = getattr(solver, "propagated_clauses", 0)

            return {
                "status": "sat" if is_sat else "unsat",
                "solve_time": solve_time,
                "is_correct": is_correct,
                "propagations": propagations,
            }
        except Exception as e:
            return {
                "status": "error",
                "solve_time": getattr(solver, "timeout", 120),
                "error": str(e),
                "propagations": 0,
            }

    def run_benchmarks(self) -> None:
        """Run all solvers on all puzzles and save results."""
        # Initialize results structure grouped by solver
        results = {
            solver_name: {
                "puzzles": {},
                "stats": {
                    "total_puzzles": 0,
                    "solved_count": 0,
                    "correct_count": 0,
                    "total_time": 0,
                    "total_propagations": 0,
                    "avg_time": 0,
                    "avg_propagations": 0,
                },
            }
            for solver_name in self.solvers
        }

        # Get all puzzle files
        puzzle_files = [f for f in os.listdir(self.puzzles_dir) if f.endswith(".json")]

        for puzzle_file in puzzle_files:
            puzzle_id = puzzle_file[:-5]  # Remove .json extension
            puzzle = self.load_puzzle(puzzle_id)
            solution = self.load_solution(puzzle_id)

            if not puzzle or not solution:
                continue

            for solver_name in self.solvers:
                print(f"Running {solver_name} on puzzle {puzzle_id}")
                result = self.run_solver(solver_name, puzzle, solution)

                # Store individual puzzle result
                results[solver_name]["puzzles"][puzzle_id] = result

                # Update solver statistics
                stats = results[solver_name]["stats"]
                stats["total_puzzles"] += 1
                if result["status"] == "sat":
                    stats["solved_count"] += 1
                if result.get("is_correct", False):
                    stats["correct_count"] += 1
                stats["total_time"] += result["solve_time"]
                stats["total_propagations"] += result["propagations"]

        # Calculate averages
        for solver_stats in results.values():
            stats = solver_stats["stats"]
            total_puzzles = stats["total_puzzles"]
            if total_puzzles > 0:
                stats["avg_time"] = stats["total_time"] / total_puzzles
                stats["avg_propagations"] = stats["total_propagations"] / total_puzzles

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.results_dir, f"benchmark_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save analysis-friendly CSV format
        csv_data = []
        for solver_name, solver_results in results.items():
            for puzzle_id, puzzle_result in solver_results["puzzles"].items():
                csv_data.append(
                    {
                        "solver": solver_name,
                        "puzzle_id": puzzle_id,
                        "status": puzzle_result["status"],
                        "solve_time": puzzle_result["solve_time"],
                        "is_correct": puzzle_result.get("is_correct", False),
                        "propagations": puzzle_result["propagations"],
                    }
                )

        csv_path = os.path.join(self.results_dir, f"benchmark_{timestamp}.csv")
        with open(csv_path, "w") as f:
            if csv_data:
                headers = csv_data[0].keys()
                f.write(",".join(headers) + "\n")
                for row in csv_data:
                    f.write(",".join(str(row[h]) for h in headers) + "\n")

        print(f"Benchmark results saved to {result_path} and {csv_path}")
