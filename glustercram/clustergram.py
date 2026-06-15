# pyright: reportExplicitAny=false

from typing import Any

from plotly.graph_objs import Figure
from glustercram.algos.distance import DistFunName
from glustercram.algos.linkage import LinkageFunName
from glustercram.dendrogram import Dendrogram
from glustercram.types import ClusteringFun, Color, DistFun, HeatmapMatrix, LayoutPoint, LinkageFun
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
        plot_bgcolor: str = "white",
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
        # [empty]      [empty]     [col. GM]     [col. GM]
        # [row dendro] [row GM]    [heatmap]     [heatmap]
        # [row dendro] [row GM]    [heatmap]     [heatmap]
        # Addressing starts from top left, row major
        rows = 4
        cols = 4
        specs: list[list[None | dict[str, Any]]] = [
            [None for _ in range(cols)] for _ in range(rows)
        ]

        # For updating layouts, rows/cols are 1-indexed row-major
        COL_DENDRO_POS = LayoutPoint(1, 3)
        ROW_DENDRO_POS = LayoutPoint(3, 1)
        COL_GM_POS = LayoutPoint(2, 3)
        ROW_GM_POS = LayoutPoint(3, 2)
        HEATMAP_POS = LayoutPoint(3, 3)

        specs[COL_DENDRO_POS.x - 1][COL_DENDRO_POS.y - 1] = {"colspan": 2}  # Column Dendrogram
        specs[ROW_DENDRO_POS.x - 1][ROW_DENDRO_POS.y - 1] = {"rowspan": 2}  # Row Dendrogram
        specs[COL_GM_POS.x - 1][COL_GM_POS.y - 1] = {"colspan": 2}  # Column Group Markers
        specs[ROW_GM_POS.x - 1][ROW_GM_POS.y - 1] = {"rowspan": 2}  # Row Group Markers
        specs[HEATMAP_POS.x - 1][HEATMAP_POS.y - 1] = {"colspan": 2, "rowspan": 2}  # Heatmap

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
            rows=[HEATMAP_POS.x] * 2,
            cols=[HEATMAP_POS.y] * 2,
        )

        
        ## DENDROGRAMS

        dendro_axes_layout = {
            "showline": False,
            "showgrid": False,
            "showticklabels": False,
        }

        def update_xyaxes(fig: Figure, subplot_pos: LayoutPoint, **kwargs):
            _ = fig.update_xaxes(row=subplot_pos.x, col=subplot_pos.y, **kwargs)
            _ = fig.update_yaxes(row=subplot_pos.x, col=subplot_pos.y, **kwargs)

        def add_tracelist(fig: Figure, subplot_pos: LayoutPoint, traces: list):
            _ = fig.add_traces(
                traces,
                rows=[subplot_pos.x] * len(traces),
                cols=[subplot_pos.y] * len(traces),
            )

        # Columns
        cols_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_cols,
            orientation="bottom",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_cols # Always use precomputed matrix
        ).data
        add_tracelist(fig, COL_DENDRO_POS, cols_dendro_traces)
        update_xyaxes(fig, COL_DENDRO_POS, **dendro_axes_layout)

        # Rows
        rows_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_rows,
            orientation="right",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_rows # Always use precomputed matrix
        ).data
        add_tracelist(fig, ROW_DENDRO_POS, rows_dendro_traces)
        update_xyaxes(fig, ROW_DENDRO_POS, **dendro_axes_layout)

        ## GROUP MARKERS
        # Rows
        # TODO

        # Columns
        # TODO


        _ = fig.update_layout(
            plot_bgcolor = plot_bgcolor,
            showlegend=False
        )

        fig.show()
        return fig
