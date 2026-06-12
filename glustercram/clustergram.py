# pyright: reportExplicitAny=false

from typing import Any
from glustercram.algos.distance import DistFunName
from glustercram.algos.linkage import LinkageFunName
from glustercram.dendrogram import Dendrogram
from glustercram.types import ClusteringFun, Color, DistFun, HeatmapMatrix, LinkageFun
from glustercram.visu.heatmap import heatmap
import glustercram.algos.distance as dist
import glustercram.algos.linkage as link
import pandas as pd
import scipy

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly import subplots


class Clustergram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance: DistFunName | DistFun,
        linkage: LinkageFunName | LinkageFun,
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

        :ivar linkage_matrix_rows: Linkage matrix for clustering of rows
        :ivar linkage_matrix_cols: Linkage matrix for clustering of columns
        :ivar permuted_data: The rearranged data for the heatmap as a 2D numpy array
        """
        self.data: pd.DataFrame = data
        self.data_rows = self.data.to_numpy()
        self.data_cols = self.data.T.to_numpy()

        """ Linkage + Distance method that performs the clustering """
        self.calc_method: ClusteringFun = link.get_preferred_implementation(
            linkage, distance
        )

        self.linkage_matrix_rows = self.calc_method(self.data_rows)
        self.linkage_matrix_cols = self.calc_method(self.data_cols)

        self.linkage_matrix_rows = scipy.cluster.hierarchy.optimal_leaf_ordering(
            self.linkage_matrix_rows, self.data_rows, dist.nan_euclidean
        )
        self.linkage_matrix_cols = scipy.cluster.hierarchy.optimal_leaf_ordering(
            self.linkage_matrix_cols, self.data_cols, dist.nan_euclidean
        )

        cols_permutation = scipy.cluster.hierarchy.leaves_list(self.linkage_matrix_cols)
        rows_permutation = scipy.cluster.hierarchy.leaves_list(self.linkage_matrix_rows)

        permuted_data = self.data_rows[rows_permutation]
        self.permuted_data: HeatmapMatrix = permuted_data[:, cols_permutation]
        self.permuted_column_labels: list[str] = [self.data.columns[int(i)] for i in cols_permutation]
        self.permuted_row_labels: list[str] = [self.data.index[int(i)] for i in rows_permutation]

    def get_visualization_plotly(
        self,
        *,
        heatmap_legend_title: str = "Heatmap legend",
        heatmap_nan_color: Color = "#000000",
        heatmap_kwargs: dict[str, Any] | None = None,
    ):
        """
        Returns the computed clustergram as a plotly figure.
        This function is based off the PROTzilla implementation of the Dash Bio clustergram
        but has been heavily refactored and adjusted

        :param heatmap_legend_title: Title for the heatmap legend shown above the color bar
        :param heatmap_nan_color: Color for heatmap cells corresponsing to NaN values
        :param heatmap_kwargs: additional kwargs passed to go.Heatmap
        """

        # GM = Group Marker
        # [empty]      [empty]     [col. dendro] [col. dendro]
        # [row dendro] [row GM]    [col. GM]     [col. GM]
        # [row dendro] [row GM]    [heatmap]     [heatmap]
        # [empty]      [empty]     [heatmap]     [heatmap]
        # Addressing starts from top left, row major
        rows = 4
        cols = 4
        specs: list[list[None | dict[str, Any]]] = [
            [None for _ in range(cols)] for _ in range(rows)
        ]

        specs[0][2] = {"colspan": 2}  # Column Dendrogram
        specs[1][0] = {"rowspan": 2}  # Row Dendrogram
        specs[1][2] = {"colspan": 2}  # Column Group Markers
        specs[1][1] = {"rowspan": 2}  # Row Group Markers
        specs[2][2] = {"colspan": 2, "rowspan": 2}  # Heatmap

        fig = subplots.make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            vertical_spacing=0.0,
            horizontal_spacing=0.0,
        )


        ## HEATMAP

        _ = fig.add_traces(
            heatmap(
                self.permuted_data,
                nan_color=heatmap_nan_color,
                heatmap_legend_title=heatmap_legend_title,
                **(heatmap_kwargs or dict()),
            ),
            rows=[2, 2],
            cols=[2, 2],
        )

        
        ## DENDROGRAMS
        # Columns
        cols_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_cols,
            orientation="bottom",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_cols # Always use precomputed matrix
        ).data

        # TODO: Add Dendro Cols to plot

        # Rows
        rows_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_rows,
            orientation="right",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_rows # Always use precomputed matrix
        ).data

        # TODO: Add Dendro Rows to plot

        ## GROUP MARKERS
        # Rows
        # TODO

        # Columns
        # TODO

        fig.show()
        return fig
