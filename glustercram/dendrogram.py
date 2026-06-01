class Dendrogram:
    """
    General dendrogram implementation. Each node has to children, which are either:
        - other dendrograms
        - indices of the data in the initial collection

    Each node is also associated with a certain height
    """

    def __init__(self, children, height: float) -> None:
        self.children: tuple[Dendrogram | int, Dendrogram | int] = children
        self.height: float = height

    def get_ordering(self) -> list[int]:
        """
        Returns a permutation of the associated data in the initial collection that corresponds
        to the dendrogram by flattening the tree
        """

        ordering: list[int] = []

        def dfs(subtree: Dendrogram):
            left_subtree = subtree.children[0]
            right_subtree = subtree.children[1]

            # Append left subtree
            if isinstance(left_subtree, int):
                ordering.append(left_subtree)
            else:
                dfs(left_subtree)

            # Append right subtree
            if isinstance(right_subtree, int):
                ordering.append(right_subtree)
            else:
                dfs(right_subtree)

        dfs(self)
        return ordering
