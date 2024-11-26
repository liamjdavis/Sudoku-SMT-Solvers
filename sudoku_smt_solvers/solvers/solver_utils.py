class SolverUtils:
    @staticmethod
    def generate_cnf(size, get_var):
        """Generate CNF clauses for 25x25 Sudoku rules"""
        clauses = []

        # Cell constraints
        for i in range(size):
            for j in range(size):
                clauses.append([get_var(i, j, k) for k in range(1, size + 1)])

                # Only one number per cell
                for k in range(1, size + 1):
                    for l in range(k + 1, size + 1):
                        clauses.append([-get_var(i, j, k), -get_var(i, j, l)])

        # Row constraints
        for i in range(size):
            for k in range(1, size + 1):
                for j in range(size):
                    for l in range(j + 1, size):
                        clauses.append([-get_var(i, j, k), -get_var(i, l, k)])

        # Column constraints
        for j in range(size):
            for k in range(1, size + 1):
                for i in range(size):
                    for l in range(i + 1, size):
                        clauses.append([-get_var(i, j, k), -get_var(l, j, k)])

        # 5x5 block constraints
        block_size = 5
        for block_i in range(block_size):
            for block_j in range(block_size):
                for k in range(1, size + 1):
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
                                    clauses.append(
                                        [
                                            -get_var(*pos1, k),
                                            -get_var(*pos2, k),
                                        ]
                                    )

        return clauses
