"""Sudoku SMT Solvers - A package for solving and benchmarking large-scale Sudoku puzzles.

This package provides various SAT and SMT-based solvers optimized for 25x25 Sudoku puzzles,
along with tools for puzzle generation and solver benchmarking.

Key Components:
- Multiple solver implementations (DPLL, DPLL(T), Z3, CVC5)
- Sudoku puzzle generator with difficulty settings
- Benchmarking suite for comparing solver performance
"""

from .solvers.dpll_solver import DPLLSolver
from .solvers.dpllt_solver import DPLLTSolver
from .solvers.z3_solver import Z3Solver
from .solvers.cvc5_solver import CVC5Solver
from .benchmarks.benchmark_runner import BenchmarkRunner
from .benchmarks.sudoku_generator.sudoku_generator import SudokuGenerator

__version__ = "0.3.0"
