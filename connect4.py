
class Board:

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.grid:list[list[str]] = [[' ' for _ in range(cols)] for _ in range(rows)]

    