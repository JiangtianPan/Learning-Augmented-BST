"""
Learning Augmented Binary Search Tree (Lin et al., 2019 ICML).

Constructs a BST based on predicted key frequencies. Keys with higher
predicted probability are placed closer to the root, minimizing the
expected weighted search cost. The construction follows a Cartesian tree
approach: among sorted keys, the one with maximum predicted frequency
becomes the root, and left/right subtrees are built recursively.
"""


class Node:
    __slots__ = ('key', 'freq', 'left', 'right', 'parent')

    def __init__(self, key, freq=0.0):
        self.key = key
        self.freq = freq
        self.left = None
        self.right = None
        self.parent = None


class LearningBST:
    """Learning Augmented BST built from predicted frequencies."""

    def __init__(self):
        self.root = None
        self._size = 0

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def build(self, keys_and_freqs):
        """Build the tree from a list of (key, predicted_frequency) pairs.

        Uses Cartesian-tree construction: sort keys, then recursively pick
        the key with the highest frequency as the subtree root.

        Args:
            keys_and_freqs: list of (key, freq) tuples.
        """
        if not keys_and_freqs:
            self.root = None
            self._size = 0
            return

        sorted_pairs = sorted(keys_and_freqs, key=lambda x: x[0])
        keys = [k for k, _ in sorted_pairs]
        freqs = [f for _, f in sorted_pairs]
        self._size = len(keys)
        self.root = self._build_recursive(keys, freqs, 0, len(keys) - 1, None)

    def _build_recursive(self, keys, freqs, lo, hi, parent):
        if lo > hi:
            return None

        # Find index with maximum frequency in [lo, hi]
        max_idx = lo
        for i in range(lo + 1, hi + 1):
            if freqs[i] > freqs[max_idx]:
                max_idx = i

        node = Node(keys[max_idx], freqs[max_idx])
        node.parent = parent
        node.left = self._build_recursive(keys, freqs, lo, max_idx - 1, node)
        node.right = self._build_recursive(keys, freqs, max_idx + 1, hi, node)
        return node

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #
    def search(self, key):
        """Search for a key. Returns (found: bool, depth: int).

        depth counts the number of comparisons (root is depth 1).
        """
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
    #  Insert (dynamic update)
    # ------------------------------------------------------------------ #
    def insert(self, key, freq):
        """Insert a key with a given predicted frequency.

        Inserts as in a normal BST, then rotates upward (like a treap)
        until the heap property on frequencies is restored.
        """
        new_node = Node(key, freq)
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
                # Key already exists; update frequency and bubble up
                current.freq = freq
                self._bubble_up(current)
                return

        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        self._size += 1

        # Bubble up to restore frequency-heap property
        self._bubble_up(new_node)

    def _bubble_up(self, node):
        """Rotate node upward while its freq exceeds its parent's freq."""
        while node.parent is not None and node.freq > node.parent.freq:
            if node == node.parent.left:
                self._rotate_right(node.parent)
            else:
                self._rotate_left(node.parent)

    # ------------------------------------------------------------------ #
    #  Delete (dynamic update)
    # ------------------------------------------------------------------ #
    def delete(self, key):
        """Delete a key from the tree.

        Finds the node, sets its frequency to -inf, rotates it down to a
        leaf, then removes it.
        """
        node = self._find_node(key)
        if node is None:
            return False

        # Push node down to a leaf by setting freq to -inf
        node.freq = float('-inf')
        while node.left is not None or node.right is not None:
            if node.left is None:
                self._rotate_left(node)
            elif node.right is None:
                self._rotate_right(node)
            elif node.left.freq >= node.right.freq:
                self._rotate_right(node)
            else:
                self._rotate_left(node)

        # Now node is a leaf; remove it
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
        """Left rotation around x: x's right child y becomes parent of x."""
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
        """Right rotation around x: x's left child y becomes parent of x."""
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
        """Return dict {key: depth} for all nodes. Root has depth 1."""
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
