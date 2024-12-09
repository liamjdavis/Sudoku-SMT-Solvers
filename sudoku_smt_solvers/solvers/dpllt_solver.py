import multiprocessing
from typing import List, Dict, Set, Optional, Tuple
from .sudoku_error import SudokuError
from .utils import generate_cnf, get_var


class DPLLTSolver:
    def __init__(self, sudoku: List[List[int]], timeout: int = 120) -> None:
        if timeout <= 0:
            raise SudokuError("Timeout must be positive")

        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = 25
        self.propagated_clauses = 0
        self.learned_clauses = []
        self.timeout = timeout
        self.clauses = []
        self.assignments = {}
        self.decision_level = 0
        self.variable_level = {}
        self.implication_graph = {}

        try:
            generate_cnf(self.size)
        except Exception as e:
            raise SudokuError(f"Error generating CNF: {e}")

    def analyze_conflict(self, conflict_clause: List[int]) -> Tuple[List[int], int]:
        """Analyze conflict and generate learned clause"""
        seen = set()
        learned_clause = []
        backtrack_level = 0

        for lit in conflict_clause:
            var = abs(lit)
            if var in self.variable_level:
                level = self.variable_level[var]
                if level == self.decision_level:
                    seen.add(var)
                elif level > 0:
                    learned_clause.append(lit)
                    backtrack_level = max(backtrack_level, level)

        return learned_clause, backtrack_level

    def propagate_theory(self) -> Optional[List[int]]:
        """Theory-specific propagation for Sudoku constraints"""
        # Check row, column and block constraints
        for i in range(self.size):
            for j in range(self.size):
                assigned_nums = set()
                for var, val in self.assignments.items():
                    if not val:
                        continue
                    row = (var - 1) // (self.size * self.size)
                    col = ((var - 1) % (self.size * self.size)) // self.size
                    num = ((var - 1) % self.size) + 1
                    if row == i and col == j:
                        if num in assigned_nums:
                            # Conflict found - return conflict clause
                            return [-get_var(i, j, num)]
                        assigned_nums.add(num)
        return None

    def dpll_t(self) -> bool:
        """Main DPLL(T) algorithm"""
        # Unit propagation
        while unit := self.find_unit_clause():
            if not self.update_clauses(unit):
                conflict_clause = [-unit]
                learned, level = self.analyze_conflict(conflict_clause)
                if not learned:
                    return False
                self.learned_clauses.append(learned)
                # Backtrack to appropriate level
                self.backtrack(level)
                continue

        # Theory propagation
        conflict = self.propagate_theory()
        if conflict:
            learned, level = self.analyze_conflict(conflict)
            if not learned:
                return False
            self.learned_clauses.append(learned)  # Add learned clause
            self.backtrack(level)
            # Don't clear learned clauses during backtracking
            return self.dpll_t()

        # Pure literal elimination
        while pure := self.find_pure_literal():
            if not self.update_clauses(pure):
                return False

        # All clauses satisfied
        if not self.clauses and not self.learned_clauses:
            return True

        # Choose variable
        var = self.choose_variable()

        # Increment decision level
        self.decision_level += 1

        # Try True
        assignments_backup = self.assignments.copy()
        clauses_backup = [clause[:] for clause in self.clauses]
        learned_backup = [clause[:] for clause in self.learned_clauses]

        if self.update_clauses(var) and self.dpll_t():
            return True

        # Try False
        self.assignments = assignments_backup
        self.clauses = clauses_backup
        self.learned_clauses = learned_backup
        if self.update_clauses(-var) and self.dpll_t():
            return True

        self.decision_level -= 1
        return False

    def backtrack(self, level: int):
        """Backtrack to specified decision level"""
        self.assignments = {
            var: val
            for var, val in self.assignments.items()
            if self.variable_level.get(var, 0) <= level
        }
        self.decision_level = level

    def choose_variable(self) -> int:
        """Choose next variable for branching using VSIDS heuristic"""
        # Simple implementation - choose first unassigned variable
        all_clauses = self.clauses + self.learned_clauses
        if all_clauses:
            return abs(all_clauses[0][0])
        return 1

    def find_unit_clause(self) -> Optional[int]:
        """Find a unit clause in the current clause set."""
        for clause in self.clauses:
            if len(clause) == 1:
                return clause[0]
        return None

    def find_pure_literal(self) -> Optional[int]:
        """Find a pure literal that appears with only one polarity in all clauses."""
        # Track positive and negative occurrences of each variable
        pos_lits = set()  # Variables appearing positively
        neg_lits = set()  # Variables appearing negatively

        # Scan all clauses including learned ones
        all_clauses = self.clauses + self.learned_clauses

        # Look for variables in unassigned clauses
        for clause in all_clauses:
            for lit in clause:
                var = abs(lit)
                # Skip if already assigned
                if var in self.assignments:
                    continue
                # Track polarity
                if lit > 0:
                    pos_lits.add(var)
                else:
                    neg_lits.add(var)

        # Find pure literals - variables that appear with only one polarity
        pure_pos = pos_lits - neg_lits  # Only positive occurrences
        pure_neg = neg_lits - pos_lits  # Only negative occurrences

        # Return first pure literal found (positive literals first)
        if pure_pos:
            return next(iter(pure_pos))
        if pure_neg:
            return -next(iter(pure_neg))

        return None

    def update_clauses(self, assignment: int) -> bool:
        """Update clauses based on new assignment"""
        var = abs(assignment)
        value = assignment > 0
        self.assignments[var] = value
        self.variable_level[var] = self.decision_level

        new_clauses = []
        new_learned = []

        # Process original clauses
        for clause in self.clauses:
            if not self.process_clause(clause, new_clauses):
                return False

        # Process learned clauses
        for clause in self.learned_clauses:
            if not self.process_clause(clause, new_learned):
                return False

        self.clauses = new_clauses
        self.learned_clauses = new_learned
        return True

    def process_clause(self, clause: List[int], new_clauses: List[List[int]]) -> bool:
        """Helper method to process a clause during update"""
        if any(
            lit > 0
            and self.assignments.get(abs(lit), False)
            or lit < 0
            and not self.assignments.get(abs(lit), True)
            for lit in clause
        ):
            self.propagated_clauses += 1
            return True

        new_clause = [lit for lit in clause if abs(lit) not in self.assignments]

        if new_clause:
            new_clauses.append(new_clause)
            return True

        self.propagated_clauses += 1
        return False

    def _solve_task(self) -> Optional[List[List[int]]]:
        """Helper method to run in separate process"""
        try:
            # Initialize with given numbers
            for i in range(self.size):
                for j in range(self.size):
                    if self.sudoku[i][j] != 0:
                        if not self.update_clauses(
                            get_var(i, j, self.sudoku[i][j], self.size)
                        ):
                            return None

            if self.dpll_t():
                return self.extract_solution()
            return None

        except Exception as e:
            raise SudokuError(f"Error solving puzzle: {e}")

    def solve(self) -> Optional[List[List[int]]]:
        """Solve the Sudoku puzzle"""
        try:
            # Skip multiprocessing if in test mode
            if hasattr(self, "_testing"):
                solution = self._solve_task()
                return solution

            # Create a process pool with 1 worker
            with multiprocessing.Pool(1) as pool:
                try:
                    # Run solver in separate process with timeout
                    async_result = pool.apply_async(self._solve_task)
                    solution = async_result.get(timeout=self.timeout)

                    return solution

                except multiprocessing.TimeoutError:
                    raise SudokuError(f"Solver timed out after {self.timeout} seconds")

        except SudokuError:
            raise
        except Exception as e:
            raise SudokuError(f"Critical solver error: {str(e)}")

        return None

    def extract_solution(self) -> List[List[int]]:
        """Extract solution from assignments"""
        solution = [[0] * self.size for _ in range(self.size)]
        for var, value in self.assignments.items():
            if value:
                row = (var - 1) // (self.size * self.size)
                col = ((var - 1) % (self.size * self.size)) // self.size
                num = ((var - 1) % self.size) + 1
                if 0 <= row < self.size and 0 <= col < self.size:
                    solution[row][col] = num
        return solution
