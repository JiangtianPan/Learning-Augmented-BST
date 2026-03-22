"""
AVL Tree — Standard balanced BST baseline.

Guarantees O(log n) worst-case search by maintaining balance factors
(|height(left) - height(right)| <= 1) at every node.
Serves as the O(log n) baseline in experiments.
"""


class Node:
    __slots__ = ('key', 'left', 'right', 'height')

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """AVL Tree (balanced BST) baseline."""

    def __init__(self):
        self.root = None
        self._size = 0

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #
    def build(self, keys):
        """Build an AVL tree by inserting keys one by one."""
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
    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            self._size += 1
            return Node(key)

        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node  # Duplicate

        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))
        return self._rebalance(node)

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #
    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            self._size -= 1
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            # Find in-order successor
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key = successor.key
            self._size += 1  # Compensate: _delete will decrement again
            node.right = self._delete(node.right, successor.key)

        if node is None:
            return None

        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))
        return self._rebalance(node)

    # ------------------------------------------------------------------ #
    #  AVL balancing
    # ------------------------------------------------------------------ #
    def _get_height(self, node):
        return node.height if node else 0

    def _get_balance(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rebalance(self, node):
        balance = self._get_balance(node)

        # Left-heavy
        if balance > 1:
            if self._get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right-heavy
        if balance < -1:
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _rotate_left(self, x):
        y = x.right
        t = y.left

        y.left = x
        x.right = t

        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _rotate_right(self, x):
        y = x.left
        t = y.right

        y.right = x
        x.left = t

        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

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
        return self._get_height(self.root)

    def __len__(self):
        return self._size

    def __contains__(self, key):
        found, _ = self.search(key)
        return found
