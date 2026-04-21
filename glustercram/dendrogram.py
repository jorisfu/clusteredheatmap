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
