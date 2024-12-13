import json
import os
import time
import multiprocessing
from typing import Dict, List, Optional

from ..solvers import CVC5Solver, DPLLSolver, DPLLTSolver, Z3Solver


class BenchmarkRunner:
    def __init__(
        self,
        puzzles_dir: str = "benchmarks/puzzles",
        results_dir: str = "benchmarks/results",
        timeout: int = 120,
    ):
        self.puzzles_dir = puzzles_dir
        self.results_dir = results_dir
        self.timeout = timeout
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

    def _solve_with_timeout(self, solver_class, puzzle, queue):
        """Helper function to run in separate process"""
        solver = solver_class(puzzle)
        result = solver.solve()
        # Pack both the result and propagation count
        queue.put((result, getattr(solver, "propagated_clauses", 0)))

    def run_solver(self, solver_name: str, puzzle: List[List[int]]) -> Dict:
        """Run a single solver on a puzzle and collect results with timeout."""
        solver_class = self.solvers[solver_name]

        # Create queue for getting results
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        # Create process for solving
        process = ctx.Process(
            target=self._solve_with_timeout, args=(solver_class, puzzle, queue)
        )

        start_time = time.time()
        process.start()
        process.join(timeout=self.timeout)

        solve_time = time.time() - start_time

        if process.is_alive():
            process.terminate()
            process.join()
            return {"status": "timeout", "solve_time": self.timeout, "propagations": 0}

        # Get result and propagation count from queue
        try:
            result, propagations = queue.get_nowait()
            return {
                "status": "sat" if result else "unsat",
                "solve_time": solve_time,
                "propagations": propagations,
            }
        except:
            return {"status": "error", "solve_time": solve_time, "propagations": 0}

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
