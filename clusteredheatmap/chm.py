# pyright: reportExplicitAny=false

from os import pread
from typing import Any

from numpy import ndarray
from clusteredheatmap.algos.distance import DistFunName
from clusteredheatmap.algos.linkage import LinkageFunName
from clusteredheatmap.types import (
    ClusteringFun,
    DistFun,
    HeatmapMatrix,
    LinkageFun,
)
import clusteredheatmap.algos.distance as dist
import clusteredheatmap.algos.linkage as link
import pandas as pd
import scipy


class ClusteredHeatMap:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        distance: DistFunName | DistFun = "euclidean",
        linkage: LinkageFunName | LinkageFun = "single",
        cluster_rows: bool = True,
        cluster_columns: bool = True,
        column_group_mappings: dict[str, dict[str, str]] | None = None,
        row_group_mappings: dict[str, dict[str, str]] | None = None,
        data_column_title: str = "Column",
        data_row_title: str = "Row",
        data_z_title: str = "Intensity",
        precomputed_dist_rows: ndarray | None = None,
        precomputed_linkage_rows: ndarray | None = None,
        precomputed_dist_cols: ndarray | None = None,
        precomputed_linkage_cols: ndarray | None = None,
    ) -> None:
        """
        Computes the necessary data for a clustered heatmap.
        To obtain a visualization after computation, use one of the get_visualization_* methods
        depending on your desired visualization tool.

        :param data: The data to cluster in pandas wide format
        :param distance: The name of the distance function to use or a custom distance function.
            Custom distance functions must be compatible with [[TODO: Signature]]
        :param linkage: The name of the linkage function to use or a custom linkage function.
            Custom linkage functions must be compatible with [[TODO: Signature]]
        :param column_group_mappings: Dicts mapping column labels to groups.
            Multiple mappings are supported, each key in this dict gets used as the respecitve
            mapping's label.
        :param row_group_mappings: Dicts mapping row labels to groups.
            Multiple mappings are supported, each key in this dict gets used as the respecitve
            mapping's label.
        :param cluster_rows: True iff clustering should be performed per-row
        :param cluster_columns: True iff clustering should be performed per-column
        :param data_column_title: Title for the data columns, i.e. what each column represents
        :param data_row_title: Title for the data rows, i.e. what each row represents
        :param data_z_title: Title for the data values, i.e. what the heat values represent
        :param precomputed_dist_rows: Condensed distance matrix for distance between rows
            in the data. Overrides calculation if given. Must be in scipy condensed distance
            matrix format (see scipy.spatial.distance.pdist docs)
        :param precomputed_dist_cols: Condensed distance matrix for distance between columns
            in the data. Overrides calculation if given. Must be in scipy condensed distance
            matrix format (see scipy.spatial.distance.pdist docs)
        :param precomputed_linkage_rows: Linkage matrix for clustering between rows in the 
            data. Overrides calculation if given. Must be in scipy linkage matrix format
            (see scipy.cluster.hierarchy.linkage docs)
        :param precomputed_linkage_columns: Linkage matrix for clustering between columns in the 
            data. Overrides calculation if given. Must be in scipy linkage matrix format
            (see scipy.cluster.hierarchy.linkage docs)

        :ivar linkage_matrix_rows: Linkage matrix for clustering of rows
        :ivar linkage_matrix_cols: Linkage matrix for clustering of columns
        :ivar permuted_data: The rearranged data for the heatmap as a 2D numpy array
        """
        self.data: pd.DataFrame = data
        self.data_rows: ndarray = self.data.to_numpy()
        self.data_cols: ndarray = self.data.T.to_numpy()

        self.cluster_rows: bool = cluster_rows
        self.cluster_columns: bool = cluster_columns

        """ Linkage + Distance method that performs the clustering """
        self.distance_method: DistFun | DistFunName = dist.get_preferred_implementation(distance)
        self.linkage_method: LinkageFun = link.get_preferred_implementation(linkage)

        cols_permutation = list(range(len(self.data_cols)))
        rows_permutation = list(range(len(self.data_rows)))

        self.distance_matrix_rows: ndarray | None = None
        self.linkage_matrix_rows: ndarray | None = None

        if self.cluster_rows:
            self.distance_matrix_rows = precomputed_dist_rows or scipy.spatial.distance.pdist(self.data_rows, metric=self.distance_method)
            self.linkage_matrix_rows = precomputed_linkage_rows or self.linkage_method(self.distance_matrix_rows)

            self.linkage_matrix_rows = scipy.cluster.hierarchy.optimal_leaf_ordering(
                self.linkage_matrix_rows, self.distance_matrix_rows
            )

            rows_permutation = scipy.cluster.hierarchy.leaves_list(
                self.linkage_matrix_rows
            )

        self.distance_matrix_cols: ndarray | None = None
        self.linkage_matrix_cols: ndarray | None = None
        if self.cluster_columns:
            self.distance_matrix_cols = precomputed_dist_cols or scipy.spatial.distance.pdist(self.data_cols, metric=self.distance_method)
            self.linkage_matrix_cols = precomputed_linkage_cols or self.linkage_method(self.distance_matrix_cols)

            self.linkage_matrix_cols = scipy.cluster.hierarchy.optimal_leaf_ordering(
                self.linkage_matrix_cols, self.distance_matrix_cols
            )

            cols_permutation = scipy.cluster.hierarchy.leaves_list(
                self.linkage_matrix_cols
            )

        permuted_data = self.data_rows[rows_permutation]
        self.permuted_data: HeatmapMatrix = permuted_data[:, cols_permutation]
        self.permuted_column_labels: list[str] = [
            self.data.columns[int(i)] for i in cols_permutation
        ]
        self.permuted_row_labels: list[str] = [
            self.data.index[int(i)] for i in rows_permutation
        ]

        self.column_group_mappings: dict[str, dict[str, str]] = (
            column_group_mappings if column_group_mappings is not None else dict()
        )
        self.row_group_mappings: dict[str, dict[str, str]] = (
            row_group_mappings if row_group_mappings is not None else dict()
        )

        self.data_row_title: str = data_row_title
        self.data_column_title: str = data_column_title
        self.data_z_title: str = data_z_title
