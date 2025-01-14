"""Sudoku SMT Solvers - A package for solving and benchmarking large-scale Sudoku puzzles.

This package provides various SAT and SMT-based solvers optimized for 25x25 Sudoku puzzles,
along with tools for puzzle generation and solver benchmarking.

Key Components:
- Multiple solver implementations (DPLL, DPLL(T), Z3, CVC5)
- Sudoku puzzle generator with difficulty settings
- Benchmarking suite for comparing solver performance
"""

from .solvers import CVC5Solver, DPLLSolver, DPLLTSolver, Z3Solver
from .solvers.utils import SudokuError
from .benchmarks import BenchmarkRunner
from .benchmarks.sudoku_generator import (
    SudokuGenerator,
    LasVegasGenerator,
    HoleDigger,
    DFSSolver,
)

__version__ = "0.4.0"

__all__ = [
    "CVC5Solver",
    "DPLLSolver",
    "DPLLTSolver",
    "Z3Solver",
    "BenchmarkRunner",
    "SudokuGenerator",
    "LasVegasGenerator",
    "HoleDigger",
    "DFSSolver",
    "SudokuError",
]
