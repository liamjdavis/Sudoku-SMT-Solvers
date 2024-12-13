# Sudoku-SMT-Solvers

[![Pytest + CI/CD](https://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/pytest-and-coverage.yml/badge.svg)](ttps://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/pytest-and-coverage.yml)
[![Coverage Status](https://coveralls.io/repos/github/liamjdavis/Sudoku-SMT-Solvers/badge.svg)](https://coveralls.io/github/liamjdavis/Sudoku-SMT-Solvers)

## About
This repository contains the code for the study "Evaluating SMT-Based Solvers on Sudoku". Created by Liam Davis (@liamjdavis) and Ryan Ji (@TairanJ) as their AI course (COSC-241) final project, it evaluates the efficacy of SMT-Based Solvers by benchmarking three modern SMT solvers (DPLL(T), Z3, and CVC5) against the DPLL algorithm on a collection of 100 25x25 Sudoku puzzles of varying difficulty.

The study aims to answer three research questions: 
1. How have logical solvers evolved over time in terms of performance and capability?
2. How do different encodings of Sudoku affect the efficiency and scalability of these solvers?
3. Are there specific features or optimizations in SMT solvers that provide a significant advantage over traditional SAT solvers for this class of problem?

## Getting started
### Installation
To run the code locally, you can install with `pip`

```bash
pip install sudoku-smt-solvers
```

### Solvers
This package includes the DPLL solver and three modern SMT solvers:
* DPLL(T)
* CVC5
* Z3

To run any of the solvers on a 25x25 Sudoku puzzle, you can create an instance of the solver class and call the solve method in a file at the root (Sudoku-smt-solvers). Here is an example using Z3:

```python
from sudoku_smt_solvers.solvers.z3_solver import Z3Solver

# Example grid (25x25)
grid = [[0] * 25 for _ in range(25)]
solver = Z3Solver(grid)
solution = solver.solve()

if solution:
    print(f"Solution:\n\n{solution}")
else:
    print("No solution exists.")
```

### Sudoku Generator
This package also includes a generator for creating Sudoku puzzles to be used as benchmarks. To generate a puzzle, create an instance of the `SudokuGenerator` class and call the `generate` method. Here is an example:

```python
from sudoku_smt_solvers.benchmarks.sudoku_generator.sudoku_generator import SudokuGenerator

generator = SudokuGenerator(size = 25, givens = 80, timeout = 5, difficulty = "Medium", puzzles_dir = "benchmarks/puzzles", solutions_dir = "benchmarks/solutions")

generator.generate()
```

Due to the computational complexity of generating large sudoku puzzles, it is recommended that you run multiple generator instances in parallel to create benchmarks.

### Benchmark Runner
To run the benchmarks you created on all four solvers, create an instance of the `BenchmarkRunner` class and call the `run_benchmarks` method. Here is an example:

```python
from sudoku_smt_solvers.benchmarks.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(
    puzzles_dir='resources/benchmarks/puzzles/',
    solutions_dir='resources/benchmarks/solutions/',
    results_dir='results/'
)
runner.run_benchmarks()
```

## Contact Us
For any questions or support, please reach out to Liam at ljdavis27 at amherst.edu and Ryan at tji26 at amherst.edu
