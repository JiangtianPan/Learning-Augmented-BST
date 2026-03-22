"""
Splay Tree (Sleator & Tarjan, 1985).

A self-adjusting BST that moves the most recently accessed node to the
root via a sequence of zig, zig-zig, and zig-zag rotations.
"""


class Node:
    __slots__ = ('key', 'left', 'right', 'parent')

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class SplayTree:
    """Splay Tree with top-down splaying."""

    def __init__(self):
        self.root = None
        self._size = 0

    # ------------------------------------------------------------------ #
    #  Splay operation
    # ------------------------------------------------------------------ #
    def _splay(self, node):
        """Splay node to the root using zig, zig-zig, zig-zag steps."""
        while node.parent is not None:
            parent = node.parent
            grandparent = parent.parent

            if grandparent is None:
                # Zig step
                if node == parent.left:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)
            elif node == parent.left and parent == grandparent.left:
                # Zig-zig (left-left)
                self._rotate_right(grandparent)
                self._rotate_right(parent)
            elif node == parent.right and parent == grandparent.right:
                # Zig-zig (right-right)
                self._rotate_left(grandparent)
                self._rotate_left(parent)
            elif node == parent.right and parent == grandparent.left:
                # Zig-zag (left-right)
                self._rotate_left(parent)
                self._rotate_right(grandparent)
            else:
                # Zig-zag (right-left)
                self._rotate_right(parent)
                self._rotate_left(grandparent)

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #
    def search(self, key):
        """Search for a key. Returns (found: bool, depth: int).

        The accessed node (or the last visited node) is splayed to the root.
        depth counts comparisons (root is depth 1).
        """
        node = self.root
        last = None
        depth = 0
        while node is not None:
            depth += 1
            last = node
            if key == node.key:
                self._splay(node)
                return True, depth
            elif key < node.key:
                node = node.left
            else:
                node = node.right

        # Splay last visited node
        if last is not None:
            self._splay(last)
        return False, depth

    # ------------------------------------------------------------------ #
    #  Insert
    # ------------------------------------------------------------------ #
    def insert(self, key):
        """Insert a key into the splay tree."""
        if self.root is None:
            self.root = Node(key)
            self._size += 1
            return

        # Standard BST insert
        parent = None
        current = self.root
        while current is not None:
            parent = current
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                # Key exists, splay it
                self._splay(current)
                return

        new_node = Node(key)
        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        self._size += 1
        self._splay(new_node)

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #
    def delete(self, key):
        """Delete a key from the splay tree."""
        node = self._find_node(key)
        if node is None:
            return False

        self._splay(node)

        if node.left is None:
            self._replace_root(node, node.right)
        elif node.right is None:
            self._replace_root(node, node.left)
        else:
            # Find in-order successor (min of right subtree)
            successor = node.right
            while successor.left is not None:
                successor = successor.left

            if successor.parent != node:
                self._replace_root(successor, successor.right)
                successor.right = node.right
                if successor.right:
                    successor.right.parent = successor

            self._replace_root(node, successor)
            successor.left = node.left
            if successor.left:
                successor.left.parent = successor

        self._size -= 1
        return True

    def _replace_root(self, old, new):
        if old.parent is None:
            self.root = new
        elif old == old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        if new is not None:
            new.parent = old.parent

    def _find_node(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    # ------------------------------------------------------------------ #
    #  Build from sorted keys
    # ------------------------------------------------------------------ #
    def build(self, keys):
        """Build a splay tree by inserting keys one by one."""
        for key in keys:
            self.insert(key)

    # ------------------------------------------------------------------ #
    #  Rotations
    # ------------------------------------------------------------------ #
    def _rotate_left(self, x):
        y = x.right
        if y is None:
            return
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        if y is None:
            return
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.right = x
        x.parent = y

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def get_all_depths(self):
        depths = {}
        self._collect_depths(self.root, 1, depths)
        return depths

    def _collect_depths(self, node, depth, depths):
        if node is None:
            return
        depths[node.key] = depth
        self._collect_depths(node.left, depth + 1, depths)
        self._collect_depths(node.right, depth + 1, depths)

    def height(self):
        return self._height(self.root)

    def _height(self, node):
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def __len__(self):
        return self._size

    def __contains__(self, key):
        found, _ = self.search(key)
        return found
