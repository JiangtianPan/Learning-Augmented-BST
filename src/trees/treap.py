"""
Treap (Seidel & Aragon, 1996).

A randomized BST where each node has a random priority. The tree
maintains BST order on keys and max-heap order on priorities.
This serves as a control baseline: if Learning BST (which uses
predicted frequencies as priorities) outperforms Treap, the
"learning" component is genuinely helpful.
"""

import random


class Node:
    __slots__ = ('key', 'priority', 'left', 'right', 'parent')

    def __init__(self, key, priority=None):
        self.key = key
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.parent = None


class Treap:
    """Treap with random priorities."""

    def __init__(self, seed=None):
        self.root = None
        self._size = 0
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, keys):
        """Build a treap by inserting keys one by one."""
        for key in keys:
            self.insert(key)

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #
    def search(self, key):
        """Search for a key. Returns (found: bool, depth: int)."""
        node = self.root
        depth = 0
        while node is not None:
            depth += 1
            if key == node.key:
                return True, depth
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return False, depth

    # ------------------------------------------------------------------ #
    #  Insert
    # ------------------------------------------------------------------ #
    def insert(self, key, priority=None):
        """Insert a key with an optional priority (random if not given)."""
        new_node = Node(key, priority)
        if self.root is None:
            self.root = new_node
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
                return  # Duplicate key

        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        self._size += 1

        # Bubble up to restore heap property
        while new_node.parent is not None and new_node.priority > new_node.parent.priority:
            if new_node == new_node.parent.left:
                self._rotate_right(new_node.parent)
            else:
                self._rotate_left(new_node.parent)

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #
    def delete(self, key):
        """Delete a key from the treap."""
        node = self._find_node(key)
        if node is None:
            return False

        # Push down to leaf
        node.priority = float('-inf')
        while node.left is not None or node.right is not None:
            if node.left is None:
                self._rotate_left(node)
            elif node.right is None:
                self._rotate_right(node)
            elif node.left.priority >= node.right.priority:
                self._rotate_right(node)
            else:
                self._rotate_left(node)

        # Remove leaf
        if node.parent is None:
            self.root = None
        elif node == node.parent.left:
            node.parent.left = None
        else:
            node.parent.right = None
        self._size -= 1
        return True

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
