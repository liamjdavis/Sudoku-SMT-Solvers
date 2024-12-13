import json
import os
import time
from typing import Dict, List, Optional

from ..solvers import CVC5Solver, DPLLSolver, DPLLTSolver, Z3Solver


class BenchmarkRunner:
    def __init__(
        self,
        puzzles_dir: str = "benchmarks/puzzles",
        results_dir: str = "benchmarks/results",
    ):
        self.puzzles_dir = puzzles_dir
        self.results_dir = results_dir
        self.solvers = {
            "CVC5": CVC5Solver,
            "DPLL": DPLLSolver,
            "DPLL(T)": DPLLTSolver,
            "Z3": Z3Solver,
        }
        os.makedirs(results_dir, exist_ok=True)

    def load_puzzle(self, puzzle_id: str) -> Optional[List[List[int]]]:
        """Load a puzzle from the puzzles directory."""
        puzzle_path = os.path.join(self.puzzles_dir, f"{puzzle_id}.json")
        try:
            with open(puzzle_path, "r") as f:
                data = json.load(f)
                for key in ["grid", "puzzle", "gridc"]:
                    if key in data:
                        return data[key]
            print(
                f"No valid grid found in {puzzle_id}. Available keys: {list(data.keys())}"
            )
            return None
        except Exception as e:
            print(f"Error loading puzzle {puzzle_id}: {e}")
            return None

    def run_solver(self, solver_name: str, puzzle: List[List[int]]) -> Dict:
        """Run a single solver on a puzzle and collect results."""
        solver_class = self.solvers[solver_name]
        solver = solver_class(puzzle)

        try:
            solution = solver.solve()
            solve_time = getattr(solver, "solve_time", getattr(solver, "timeout", 120))
            is_sat = solution is not None
            propagations = getattr(solver, "propagated_clauses", 0)

            return {
                "status": "sat" if is_sat else "unsat",
                "solve_time": solve_time,
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
        results = {
            solver_name: {
                "puzzles": {},
                "stats": {
                    "total_puzzles": 0,
                    "solved_count": 0,
                    "total_time": 0,
                    "total_propagations": 0,
                    "avg_time": 0,
                    "avg_propagations": 0,
                },
            }
            for solver_name in self.solvers
        }

        puzzle_files = [f for f in os.listdir(self.puzzles_dir) if f.endswith(".json")]
        print(f"Found {len(puzzle_files)} puzzle files")  # Debug

        for puzzle_file in puzzle_files:
            puzzle_id = puzzle_file[:-5]
            puzzle = self.load_puzzle(puzzle_id)

            if not puzzle:
                print(f"Failed to load puzzle: {puzzle_id}")  # Debug
                continue

            for solver_name in self.solvers:
                print(f"Running {solver_name} on puzzle {puzzle_id}")
                result = self.run_solver(solver_name, puzzle)
                print(f"Result: {result}")  # Debug

                results[solver_name]["puzzles"][puzzle_id] = result

                stats = results[solver_name]["stats"]
                stats["total_puzzles"] += 1
                if result["status"] == "sat":
                    stats["solved_count"] += 1
                stats["total_time"] += result["solve_time"]
                stats["total_propagations"] += result["propagations"]

        # Calculate averages
        for solver_name, solver_stats in results.items():
            stats = solver_stats["stats"]
            total_puzzles = stats["total_puzzles"]
            if total_puzzles > 0:
                stats["avg_time"] = stats["total_time"] / total_puzzles
                stats["avg_propagations"] = stats["total_propagations"] / total_puzzles
            print(f"Stats for {solver_name}: {stats}")  # Debug

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.results_dir, f"benchmark_{timestamp}.json")

        # Debug CSV data
        csv_data = []
        for solver_name, solver_results in results.items():
            for puzzle_id, puzzle_result in solver_results["puzzles"].items():
                row = {
                    "solver": solver_name,
                    "puzzle_id": puzzle_id,
                    "status": puzzle_result["status"],
                    "solve_time": puzzle_result["solve_time"],
                    "propagations": puzzle_result["propagations"],
                }
                csv_data.append(row)
                print(f"Adding CSV row: {row}")  # Debug

        csv_path = os.path.join(self.results_dir, f"benchmark_{timestamp}.csv")
        print(f"Writing {len(csv_data)} rows to CSV")  # Debug

        with open(csv_path, "w") as f:
            if csv_data:
                headers = csv_data[0].keys()
                f.write(",".join(headers) + "\n")
                for row in csv_data:
                    f.write(",".join(str(row[h]) for h in headers) + "\n")

        print(f"Benchmark results saved to {result_path} and {csv_path}")
