# Sudoku-SMT-Solvers

[![Pytest + CI/CD](https://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/test.yml/badge.svg)](ttps://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/liamjdavis/Sudoku-SMT-Solvers/badge.svg)](https://coveralls.io/github/liamjdavis/Sudoku-SMT-Solvers)
[![Docs Build Deployment](https://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/docs.yml/badge.svg)](https://github.com/liamjdavis/Sudoku-SMT-Solvers/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://liamjdavis.github.io/Sudoku-SMT-Solvers)


## About
This repository contains the code for the study "Evaluating SMT-Based Solvers on Sudoku". Created by Liam Davis (@liamjdavis) and Tairan "Ryan" Ji (@TairanJ) as their for COSC-241 Artificial Intelligence at Amherst College, it evaluates the efficacy of SMT-Based Solvers by benchmarking three modern SMT solvers (DPLL(T), Z3, and CVC5) against the DPLL algorithm on a collection of 100 25x25 Sudoku puzzles of varying difficulty.

Along with the study, we also published `sudoku-smt-solvers`, a Python package that provides the various SMT-based Sudoku solvers and benchmarking tools we built for this study. The package features DPLL(T), Z3, and CVC5 solvers optimized for 25x25 Sudoku puzzles, a puzzle generator for creating test cases, and a comprehensive benchmarking suite. Available through pip, it offers a simple API for solving Sudoku puzzles using state-of-the-art SMT solvers while facilitating performance comparisons between different solving approaches.

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
from sudoku_smt_solvers import Z3Solver

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
from sudoku_smt_solvers import SudokuGenerator

generator = SudokuGenerator(size = 25, givens = 80, timeout = 5, difficulty = "Medium", puzzles_dir = "benchmarks/puzzles", solutions_dir = "benchmarks/solutions")

generator.generate()
```

Due to the computational complexity of generating large sudoku puzzles, it is recommended that you run multiple generator instances in parallel to create benchmarks.

### Benchmark Runner
To run the benchmarks you created on all four solvers, create an instance of the `BenchmarkRunner` class and call the `run_benchmarks` method. Here is an example:

```python
from sudoku_smt_solvers import BenchmarkRunner

runner = BenchmarkRunner(
    puzzles_dir='resources/benchmarks/puzzles/',
    results_dir='results/'
)
runner.run_benchmarks()
```

## Contributing

We welcome contributions in the form of new solvers, additions to our benchmark suite, or anything that improves the tool! Here's how to get started:

### Development Setup

1. **Fork and Clone**:  
   Begin by forking the repository and cloning your fork locally:
   ```bash
   git clone https://github.com/yourusername/Sudoku-SMT-Solvers.git
   cd Sudoku-SMT-Solvers
   ```

2. **Create and Activate a Virtual Environment**:  
   Set up a Python virtual environment to isolate your dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:  
   Install the required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Pre-Commit Hooks**:  
   Install and configure pre-commit hooks to maintain code quality:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

   To manually run the hooks and verify code compliance, use:
   ```bash
   pre-commit run
   ```

5. **Testing and Coverage Requirements**:  
   - Write tests for any new code or modifications.
   - Use `pytest` for running tests:
     ```bash
     pytest
     ```
   - Ensure the test coverage is at least 90%:

6. **Add and Commit Your Changes**:  
   - Follow the existing code style and structure.
   - Verify that all pre-commit hooks pass and the test coverage meets the minimum requirement.
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

7. **Push Your Branch**:
   Push your changes to your forked repository:
   ```bash
   git push origin your-branch-name
   ```

8. **Open a PR for us to review**
---

Thank you for your interest in contributing to Sudoku-SMT-Solvers! Your efforts help make this project better for everyone.


## Contact Us
For any questions or support, please reach out to Liam at ljdavis27 at amherst.edu and Ryan at tji26 at amherst.edu
