from .sudoku_error import SudokuError
import signal


class DPLLSolver:
    def __init__(self, sudoku, timeout=120) -> None:
        if not sudoku or not isinstance(sudoku, list) or len(sudoku) != 25:
            raise SudokuError("Invalid Sudoku puzzle: must be a 25x25 grid")

        self.sudoku = sudoku
        self.size = 25
        self.propagated_clauses = 0
        self.timeout = timeout
        self.clauses = []
        self.assignments = {}

        # Generate CNF Form of Sudoku
        try:
            self.generate_cnf()
        except Exception as e:
            raise SudokuError(f"Error generating CNF: {e}")

    def generate_cnf(self):
        """Generate CNF clauses for 25x25 Sudoku rules"""
        # Cell constraints
        for i in range(self.size):
            for j in range(self.size):
                self.clauses.append(
                    [self.get_var(i, j, k) for k in range(1, self.size + 1)]
                )

                # Only one number per cell
                for k in range(1, self.size + 1):
                    for l in range(k + 1, self.size + 1):
                        self.clauses.append(
                            [-self.get_var(i, j, k), -self.get_var(i, j, l)]
                        )

        # Row constraints
        for i in range(self.size):
            for k in range(1, self.size + 1):
                for j in range(self.size):
                    for l in range(j + 1, self.size):
                        self.clauses.append(
                            [-self.get_var(i, j, k), -self.get_var(i, l, k)]
                        )

        # Column constraints
        for j in range(self.size):
            for k in range(1, self.size + 1):
                for i in range(self.size):
                    for l in range(i + 1, self.size):
                        self.clauses.append(
                            [-self.get_var(i, j, k), -self.get_var(l, j, k)]
                        )

        # 5x5 block constraints
        block_size = 5
        for block_i in range(block_size):
            for block_j in range(block_size):
                for k in range(1, self.size + 1):
                    for i in range(block_size):
                        for j in range(block_size):
                            for i2 in range(i, block_size):
                                for j2 in range(j + 1 if i2 == i else 0, block_size):
                                    pos1 = (
                                        block_i * block_size + i,
                                        block_j * block_size + j,
                                    )
                                    pos2 = (
                                        block_i * block_size + i2,
                                        block_j * block_size + j2,
                                    )
                                    self.clauses.append(
                                        [
                                            -self.get_var(*pos1, k),
                                            -self.get_var(*pos2, k),
                                        ]
                                    )

    def get_var(self, row, col, num):
        """Convert Sudoku position and number to CNF variable"""
        return row * self.size * self.size + col * self.size + num

    def find_unit_clause(self):
        """Find a unit clause in the current formula"""
        for clause in self.clauses:
            unassigned = []
            for lit in clause:
                if abs(lit) not in self.assignments:
                    unassigned.append(lit)
                elif (lit > 0 and self.assignments[abs(lit)]) or (
                    lit < 0 and not self.assignments[abs(lit)]
                ):
                    # Clause is satisfied
                    break
            else:
                if len(unassigned) == 1:
                    return unassigned[0]
        return None

    def find_pure_literal(self):
        """Find a pure literal in the current formula"""
        # Track both positive and negative occurrences of each variable
        occurrences = {}

        for clause in self.clauses:
            if any(var in self.assignments.keys() for var in clause):
                continue

            for lit in clause:
                var = abs(lit)
                if var not in self.assignments:  # Ignore already assigned variables
                    if var not in occurrences:
                        occurrences[var] = {"pos": False, "neg": False}
                    if lit > 0:
                        occurrences[var]["pos"] = True
                    else:
                        occurrences[var]["neg"] = True

        # Find a variable that appears only positively or only negatively
        for var, counts in occurrences.items():
            if counts["pos"] and not counts["neg"]:  # Purely positive literal
                return var
            if counts["neg"] and not counts["pos"]:  # Purely negative literal
                return -var

        return None

    def update_clauses(self, assignment):
        """Update clauses based on new assignment"""
        var = abs(assignment)
        value = assignment > 0
        self.assignments[var] = value

        new_clauses = []

        for clause in self.clauses:
            if any(
                lit > 0
                and self.assignments.get(abs(lit), False)
                or lit < 0
                and not self.assignments.get(abs(lit), True)
                for lit in clause
            ):
                self.propagated_clauses += 1

                continue

            new_clause = [lit for lit in clause if abs(lit) not in self.assignments]

            if new_clause:
                new_clauses.append(new_clause)

            elif not new_clause:
                self.propagated_clauses += 1
                return False

        self.clauses = new_clauses

        return True

    def dpll(self):
        """Main DPLL algorithm"""
        # Unit propagation
        while unit := self.find_unit_clause():
            if not self.update_clauses(unit):
                return False

        # Pure literal elimination
        while pure := self.find_pure_literal():
            if not self.update_clauses(pure):
                return False

        # All clauses satisfied
        if not self.clauses:
            return True

        # Choose variable
        var = abs(self.clauses[0][0])

        # Try True
        assignments_backup = self.assignments.copy()
        clauses_backup = [clause[:] for clause in self.clauses]
        if self.update_clauses(var) and self.dpll():
            return True

        # Try False
        self.assignments = assignments_backup
        self.clauses = clauses_backup
        if self.update_clauses(-var) and self.dpll():
            return True

        return False

    def timeout_handler(self, signum, frame):
        raise TimeoutError("DPLL solver timed out")

    def solve(self):
        """Solve the Sudoku puzzle"""
        try:
            for i in range(self.size):
                for j in range(self.size):
                    if self.sudoku[i][j] != 0:
                        if not self.update_clauses(
                            self.get_var(i, j, self.sudoku[i][j])
                        ):
                            return None

            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(self.timeout)

            try:
                if self.dpll():
                    solution = [[0] * self.size for _ in range(self.size)]
                    for var, value in self.assignments.items():
                        if value:
                            row = (var - 1) // (self.size * self.size)
                            col = ((var - 1) % (self.size * self.size)) // self.size
                            num = ((var - 1) % self.size) + 1
                            if 0 <= row < self.size and 0 <= col < self.size:
                                solution[row][col] = num
                    return solution
            finally:
                signal.alarm(0)
        except TimeoutError:
            raise SudokuError(f"Solver timed out after {self.timeout} seconds")
        except Exception as e:
            raise SudokuError(f"Error solving puzzle: {e}")

        return None
