class SudokuError(Exception):
    """An exception class for errors in the sudoku solver"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
