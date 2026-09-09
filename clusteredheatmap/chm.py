# pyright: reportExplicitAny=false
from typing import Any

from numpy import ndarray
from clusteredheatmap.algos.distance import DistFunName
from clusteredheatmap.algos.linkage import LinkageFunName
from clusteredheatmap.types import (
    DistFun,
    HeatmapMatrix,
    LinkageFun,
    PDistFun,
)
import clusteredheatmap.algos.distance as dist
import clusteredheatmap.algos.linkage as link
import pandas as pd
import scipy

DistSpec = DistFunName | DistFun
DistArgs = dict[str, Any] | None

LinkageSpec = LinkageFunName | LinkageFun

class ClusteredHeatMap:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        distance: DistSpec | tuple[DistSpec, DistSpec] = "euclidean",
        distance_args: DistArgs | tuple[DistArgs, DistArgs] = None,
        use_completecase_analysis: bool | tuple[bool, bool] = False,
        linkage: LinkageSpec | tuple[LinkageSpec, LinkageSpec] = "single",
        cluster_rows: bool = True,
        cluster_columns: bool = True,
        column_group_mappings: dict[str, dict[str, str]] | None = None,
        row_group_mappings: dict[str, dict[str, str]] | None = None,
        data_column_title: str = "Column",
        data_row_title: str = "Row",
        data_z_title: str = "Intensity",
        distance_matrix_rows: ndarray | None = None,
        linkage_matrix_rows: ndarray | None = None,
        distance_matrix_cols: ndarray | None = None,
        linkage_matrix_cols: ndarray | None = None,
        optimal_leaf_ordering: bool = True,
        is_symmetric: bool = False,
    ) -> None:
        """
        Computes the necessary data for a clustered heatmap.
        To obtain a visualization after computation, use one of the get_visualization_* methods
        depending on your desired visualization tool.

        :param data: The data to cluster in pandas wide format
        :param distance: The name of the distance function to use or a custom distance function.
            Custom distance functions must be compatible with scipy.spatial.distance functions
            betweeen two vectors.
            Different distance functions for rows and columns can be selected by passing
            a tuple, specifying row distance at index 0 and column distance at 1.
        :param distance_args: Additional arguments passed to the distance function.
            Only applied if distance function name is given, passed callables
            must have their additional arguments hardcoded (e.g. with a lambda).
            Different arguments for row/col distance are passed as a tuple (see distance param).
        :param use_completecase_analysis: Whether or not to use complete case
            analysis for distance between vectors (only use features with pairwise
            completeness without any adjustment). Generally NOT RECOMMENDED.
            Only applies to passed distance functions and scipy provided functions.
            Can be passed as tuple for separate row/col setting (see distance param).
        :param linkage: The name of the linkage function to use or a custom linkage function.
            Custom linkage functions must be compatible with scipy.cluster.hierarchy.linkage.
            Different linkage functions for rows and columns can be selected by passing
            a tuple, specifying row linkage at index 0 and column linkage at 1.
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
        :param distance_matrix_rows: Condensed distance matrix for distance between rows
            in the data. Overrides calculation if given. Must be in scipy condensed distance
            matrix format (see scipy.spatial.distance.pdist docs)
        :param distance_matrix_cols: Condensed distance matrix for distance between columns
            in the data. Overrides calculation if given. Must be in scipy condensed distance
            matrix format (see scipy.spatial.distance.pdist docs)
        :param linkage_matrix_rows: Linkage matrix for clustering between rows in the
            data. Overrides calculation if given. Must be in scipy linkage matrix format
            (see scipy.cluster.hierarchy.linkage docs)
        :param linkage_matrix_cols: Linkage matrix for clustering between columns in the
            data. Overrides calculation if given. Must be in scipy linkage matrix format
            (see scipy.cluster.hierarchy.linkage docs)
        :param optimal_leaf_ordering: Whether or not to use optimal leaf ordering for
            the dendrograms.
        :param is_symmetric: Whether or not the input data should be interpreted as 
            a symmetric matrix. Requires cluster_rows and cluster_columns to be True.

        :ivar linkage_matrix_rows: Linkage matrix for clustering of rows
        :ivar linkage_matrix_cols: Linkage matrix for clustering of columns
        :ivar permuted_data: The rearranged data for the heatmap as a 2D numpy array
        """
        self.data: pd.DataFrame = data
        self.data_rows: ndarray = self.data.to_numpy()
        self.data_cols: ndarray = self.data.T.to_numpy()

        self.cluster_rows: bool = cluster_rows
        self.cluster_columns: bool = cluster_columns

        if is_symmetric:
            if not (cluster_rows and cluster_columns):
                raise ValueError("Cannot disable clustering on any axis for symmetric inputs.")

            if not self.data_rows.shape[0] == self.data_rows.shape[1]:
                raise ValueError("Input cannot be interpreted as symmetric since matrix is not square.")

        if isinstance(distance, tuple):
            # Propagate no distance args, avoid annoying error
            if distance_args is None:
                distance_args = (None, None)
            assert isinstance(distance_args, tuple), "Distance parameters must be consistent between tuples and non-tuples."
            assert isinstance(use_completecase_analysis, tuple), "Distance parameters must be consistent between tuples and non-tuples."
            self.pdist_method_rows: PDistFun = dist.get_preferred_pdist_implementation(
                distance[0], distance_args[0], use_completecase_analysis[0]
            )
            self.pdist_method_cols: PDistFun = dist.get_preferred_pdist_implementation(
                distance[1], distance_args[1], use_completecase_analysis[1]
            )

        else:
            assert not isinstance(distance_args, tuple), "Distance parameters must be consistent between tuples and non-tuples."
            assert not isinstance(use_completecase_analysis, tuple), "Distance parameters must be consistent between tuples and non-tuples."
            self.pdist_method_rows: PDistFun = dist.get_preferred_pdist_implementation(
                distance, distance_args, use_completecase_analysis
            )
            self.pdist_method_cols: PDistFun = self.pdist_method_rows

        if isinstance(linkage, tuple):
            self.linkage_method_rows: LinkageFun = link.get_preferred_implementation(linkage[0])
            self.linkage_method_cols: LinkageFun = link.get_preferred_implementation(linkage[1])
        else:
            self.linkage_method_rows: LinkageFun = link.get_preferred_implementation(linkage)
            self.linkage_method_cols: LinkageFun = self.linkage_method_rows

        cols_permutation = list(range(len(self.data_cols)))
        rows_permutation = list(range(len(self.data_rows)))

        self.distance_matrix_rows: ndarray | None = distance_matrix_rows
        self.linkage_matrix_rows: ndarray | None = linkage_matrix_rows
        if self.cluster_rows:
            if self.distance_matrix_rows is None:
                self.distance_matrix_rows = self.pdist_method_rows(self.data_rows)

            if self.linkage_matrix_rows is None:
                self.linkage_matrix_rows = self.linkage_method_rows(
                    self.distance_matrix_rows
                )

            if optimal_leaf_ordering:
                self.linkage_matrix_rows = (
                    scipy.cluster.hierarchy.optimal_leaf_ordering(
                        self.linkage_matrix_rows, self.distance_matrix_rows
                    )
                )

            rows_permutation = scipy.cluster.hierarchy.leaves_list(
                self.linkage_matrix_rows
            )

        self.distance_matrix_cols: ndarray | None = distance_matrix_cols
        self.linkage_matrix_cols: ndarray | None = linkage_matrix_cols
        if self.cluster_columns:

            if is_symmetric:
                self.distance_matrix_cols = self.distance_matrix_rows.copy()
                self.linkage_matrix_cols = self.linkage_matrix_rows.copy()

            else:
                if self.distance_matrix_cols is None:
                    self.distance_matrix_cols = self.pdist_method_cols(self.data_cols)

                if self.linkage_matrix_cols is None:
                    self.linkage_matrix_cols = self.linkage_method_cols(
                        self.distance_matrix_cols
                    )

                if optimal_leaf_ordering:
                    self.linkage_matrix_cols = (
                        scipy.cluster.hierarchy.optimal_leaf_ordering(
                            self.linkage_matrix_cols, self.distance_matrix_cols
                        )
                    )

            cols_permutation = scipy.cluster.hierarchy.leaves_list(
                self.linkage_matrix_cols
            )

        permuted_data = self.data_rows[rows_permutation]
        self.permuted_data: HeatmapMatrix = permuted_data[:, cols_permutation]
        self.permuted_column_labels: list[str] = [
            str(self.data.columns[int(i)]) for i in cols_permutation
        ]
        self.permuted_row_labels: list[str] = [
            str(self.data.index[int(i)]) for i in rows_permutation
        ]

        self.column_group_mappings: dict[str, dict[str, str]] = (
            column_group_mappings if column_group_mappings is not None else dict()
        )
        self.row_group_mappings: dict[str, dict[str, str]] = (
            row_group_mappings if row_group_mappings is not None else dict()
        )

        self.permuted_row_fulldescriptions: list[str] = (
            stringify_labels_with_group_mappings(
                self.permuted_row_labels, self.row_group_mappings
            )
        )
        self.permuted_col_fulldescriptions: list[str] = (
            stringify_labels_with_group_mappings(
                self.permuted_column_labels, self.column_group_mappings
            )
        )

        self.data_row_title: str = data_row_title
        self.data_column_title: str = data_column_title
        self.data_z_title: str = data_z_title


def stringify_labels_with_group_mappings(
    labels: list[str], mappings: dict[str, dict[str, str]]
):
    strings: list[str] = []
    for label in labels:
        groups = ""
        for group_name, mapping in mappings.items():
            group = mapping.get(label)
            if group is not None:
                groups += group_name + ": " + group + ", "
        if len(groups) > 0:
            groups = groups[:-2]
            groups = " (" + groups + ")"

        strings.append(label + groups)

    return strings
