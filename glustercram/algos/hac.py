from collections.abc import Iterable

from numpy.typing import NDArray
from glustercram.dendrogram import Dendrogram
from glustercram.types import T, DistFun, LinkageFun

import numpy as np

inf = float("inf")


def general(data: list[T], distance: DistFun[T], linkage: LinkageFun[T]) -> Dendrogram:
    """
    Trivial O(n³) algorithm for HAC as described in Section 8.2.1
    TODO: Full citation
    """
    n = len(data)

    # Calculate initial distance matrix and minimum distance
    prev_dist_mat_labels: list[Dendrogram | int] = list(range(n))
    prev_dist_mat = np.zeros((n, n))
    min_dist_indices = (0, 1)

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(data[i], data[j])
            prev_dist_mat[i, j] = d
            if d < prev_dist_mat[min_dist_indices]:
                min_dist_indices = (i, j)

    print(min_dist_indices)
    print(prev_dist_mat)

    for dim in range(n - 1, 0, -1):
        # Create new cluster from previous min distance
        height: float = float(n - dim)  # TODO
        cluster = Dendrogram(
            (
                prev_dist_mat_labels[min_dist_indices[0]],
                prev_dist_mat_labels[min_dist_indices[1]],
            ),
            height,
        )

        # Move cluster to the left and update labels
        dist_mat_labels = [cluster] + [
            label for label in prev_dist_mat_labels if label not in cluster.children
        ]

        # New distance matrix (one dimension less)
        dist_mat = np.zeros((dim, dim))

        # Map from current row/column index to corresponding index in previous matrix
        # Required to obtain old values
        idx_map = [None] + [
            idx
            for idx, i in enumerate(prev_dist_mat_labels)
            if i not in cluster.children
        ]

        min_dist_indices = (0, 1)

        # Update first row with linkage criterion
        for col in range(1, dim):
            d = min(
                prev_dist_mat[min_dist_indices[0], idx_map[col]],
                prev_dist_mat[min_dist_indices[1], idx_map[col]],
            )
            dist_mat[0, col] = d

            # Update potential lowest distance
            if d < dist_mat[min_dist_indices]:
                min_dist_indices = (0, col)

        # Keep remaining rows
        for row in range(1, dim):
            for col in range(row + 1, dim):
                d = prev_dist_mat[idx_map[row], idx_map[col]]
                dist_mat[row, col] = d

                # Update potential lowest distance
                if d < dist_mat[min_dist_indices]:
                    min_dist_indices = (row, col)

        # Shift tables
        prev_dist_mat_labels = dist_mat_labels
        prev_dist_mat = dist_mat

        print(dist_mat)

    #
    # print(dist_mat)
    # print(np.triu(np.ones((n, n))))

    # for _ in range(n - 1):
    # np.min(dist_mat, where=)

    return Dendrogram([1, 1], 0)
