"""
Algorithm X with Dancing Links (DLX) — Knuth's exact cover solver.

The data structure is a toroidal doubly-linked list of 1-nodes organized
into columns. Each column represents a constraint; each row represents
a choice that satisfies some subset of constraints.
"""


class Node:
    """A node in the Dancing Links structure."""
    __slots__ = ('left', 'right', 'up', 'down', 'column', 'row_id')

    def __init__(self, column=None, row_id=None):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = column
        self.row_id = row_id


class Column(Node):
    """A column header node, tracking the column size."""
    __slots__ = ('size', 'name')

    def __init__(self, name=None):
        super().__init__()
        self.column = self
        self.size = 0
        self.name = name


class DLX:
    """Exact cover solver using Dancing Links.

    Usage:
        dlx = DLX(column_names)
        dlx.add_row(row_id, column_indices_or_names)
        ...
        solutions = dlx.solve()  # list of lists of row_ids
    """

    def __init__(self, columns):
        """Initialize the DLX structure.

        Args:
            columns: list of column names/identifiers
        """
        self.header = Column("header")
        self.columns = {}  # name -> Column node
        self._col_list = []  # ordered list for index-based access
        self.solution = []
        self.solutions = []

        # Build column headers linked left-right
        prev = self.header
        for name in columns:
            col = Column(name)
            self.columns[name] = col
            self._col_list.append(col)
            # Insert col to the left of header (i.e., at end of row)
            col.right = self.header
            col.left = prev
            prev.right = col
            self.header.left = col
            prev = col

    def add_row(self, row_id, cols):
        """Add a row to the matrix.

        Args:
            row_id: identifier for this row (returned in solutions)
            cols: list of column names or indices that this row covers
        """
        first = None
        prev = None
        for c in cols:
            if isinstance(c, int):
                col = self._col_list[c]
            else:
                col = self.columns[c]

            node = Node(column=col, row_id=row_id)

            # Insert into column (above the header = at bottom of column)
            node.down = col
            node.up = col.up
            col.up.down = node
            col.up = node
            col.size += 1

            if first is None:
                first = node
                prev = node
            else:
                # Link left-right within the row
                node.left = prev
                node.right = first
                prev.right = node
                first.left = node
                prev = node

    def _cover(self, col):
        """Cover a column: remove it and all rows that intersect it."""
        col.right.left = col.left
        col.left.right = col.right
        row = col.down
        while row is not col:
            j = row.right
            while j is not row:
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1
                j = j.right
            row = row.down

    def _uncover(self, col):
        """Uncover a column: restore it and all rows that intersect it."""
        row = col.up
        while row is not col:
            j = row.left
            while j is not row:
                j.column.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            row = row.up
        col.right.left = col
        col.left.right = col

    def _choose_column(self):
        """Choose the column with the minimum size (MRV heuristic)."""
        min_size = float('inf')
        best = None
        col = self.header.right
        while col is not self.header:
            if col.size < min_size:
                min_size = col.size
                best = col
                if min_size == 0:
                    break  # Can't do better; this will cause backtracking
            col = col.right
        return best

    def solve(self, find_all=False):
        """Run Algorithm X to find exact cover solutions.

        Args:
            find_all: if True, find all solutions; otherwise stop at first

        Returns:
            list of solutions, where each solution is a list of row_ids
        """
        self.solutions = []
        self.solution = []
        self._search(find_all)
        return self.solutions

    def _search(self, find_all):
        """Recursive Algorithm X search."""
        if self.header.right is self.header:
            # All columns covered — found a solution
            self.solutions.append(list(self.solution))
            return True

        col = self._choose_column()
        if col.size == 0:
            return False  # Dead end

        self._cover(col)

        row = col.down
        while row is not col:
            self.solution.append(row.row_id)

            # Cover all other columns in this row
            j = row.right
            while j is not row:
                self._cover(j.column)
                j = j.right

            found = self._search(find_all)
            if found and not find_all:
                self._uncover_row(row)
                self._uncover(col)
                return True

            # Undo: uncover columns in reverse order
            self.solution.pop()
            self._uncover_row(row)

            row = row.down

        self._uncover(col)
        return False

    def _uncover_row(self, row):
        """Uncover all columns covered by a row (in reverse order)."""
        j = row.left
        while j is not row:
            self._uncover(j.column)
            j = j.left
