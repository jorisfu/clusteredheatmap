# pyright: reportExplicitAny=false

from typing import Any

from plotly.graph_objs import Figure
from glustercram.algos.distance import DistFunName
from glustercram.algos.linkage import LinkageFunName
from glustercram.dendrogram import Dendrogram
from glustercram.types import (
    ClusteringFun,
    Color,
    DistFun,
    HeatmapMatrix,
    LayoutPoint,
    LinkageFun,
)
from glustercram.visu.heatmap import heatmap
import glustercram.algos.distance as dist
import glustercram.algos.linkage as link
import pandas as pd
import numpy as np
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
        column_group_mapping: dict[str, str] | None = None,
        row_group_mapping: dict[str, str] | None = None,
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
        :param column_group_mapping: Dict mapping columns to groups
        :param row_group_mapping: Dict mapping columns to groups

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
        self.permuted_column_labels: list[str] = [
            self.data.columns[int(i)] for i in cols_permutation
        ]
        self.permuted_row_labels: list[str] = [
            self.data.index[int(i)] for i in rows_permutation
        ]

        self.column_group_mapping: dict[str, str] = (
            column_group_mapping if column_group_mapping is not None else dict()
        )
        self.row_group_mapping: dict[str, str] = (
            row_group_mapping if row_group_mapping is not None else dict()
        )

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

        specs[COL_DENDRO_POS.x - 1][COL_DENDRO_POS.y - 1] = {
            "colspan": 2
        }  # Column Dendrogram
        specs[ROW_DENDRO_POS.x - 1][ROW_DENDRO_POS.y - 1] = {
            "rowspan": 2
        }  # Row Dendrogram
        specs[COL_GM_POS.x - 1][COL_GM_POS.y - 1] = {
            "colspan": 2
        }  # Column Group Markers
        specs[ROW_GM_POS.x - 1][ROW_GM_POS.y - 1] = {"rowspan": 2}  # Row Group Markers
        specs[HEATMAP_POS.x - 1][HEATMAP_POS.y - 1] = {
            "colspan": 2,
            "rowspan": 2,
        }  # Heatmap

        fig = subplots.make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            vertical_spacing=0.0,
            horizontal_spacing=0.0,
        )

        def update_xyaxes(fig: Figure, subplot_pos: LayoutPoint, **kwargs):
            _ = fig.update_xaxes(row=subplot_pos.x, col=subplot_pos.y, **kwargs)
            _ = fig.update_yaxes(row=subplot_pos.x, col=subplot_pos.y, **kwargs)

        def add_tracelist(fig: Figure, subplot_pos: LayoutPoint, traces: list):
            _ = fig.add_traces(
                traces,
                rows=[subplot_pos.x] * len(traces),
                cols=[subplot_pos.y] * len(traces),
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

        update_xyaxes(fig, HEATMAP_POS, showticklabels=False)

        ## DENDROGRAMS

        dendro_axes_layout = {
            "showline": False,
            "showgrid": False,
            "showticklabels": False,
        }

        # Columns
        cols_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_cols,
            orientation="bottom",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_cols,  # Always use precomputed matrix
        ).data
        add_tracelist(fig, COL_DENDRO_POS, cols_dendro_traces)
        update_xyaxes(fig, COL_DENDRO_POS, **dendro_axes_layout)

        # Rows
        rows_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_rows,
            orientation="right",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_rows,  # Always use precomputed matrix
        ).data
        add_tracelist(fig, ROW_DENDRO_POS, rows_dendro_traces)
        update_xyaxes(fig, ROW_DENDRO_POS, **dendro_axes_layout)

        ## GROUP MARKERS
        def create_group_marker_trace(
            data_labels: list[str],
            label_to_group: dict[str, str],
            group_to_color: dict[str, Color],
            default_color: str = "#000000",
            is_vertical: bool = False,
        ):
            """
            Creates a trace used for group markers.

            :param data_labels: The labels of the data as ordered on the axis to mark
            :param label_to_group: Mapping of data labels to group identifiers.
                If incomplete, mapping is performed to None
            :param group_to_color: Mapping of group identifiers to colors.
                If incomplete, mapping is performed to a default color scale TODO: Actually just this
            :param default_color: Color for data points without an associated group
            :param is_vertical: True iff the group markers are to be arranged vertically
            """

            all_groups = list(set(label_to_group.values()))
            amount_of_groups = len(all_groups)

            # Embedding like this required for continuous colorscale
            group_to_z: dict[str | None, float] = {
                g: i + 0.5
                for i, g in enumerate(all_groups)  # Center within bin of size 1
            }

            data_as_groups = [label_to_group.get(label) for label in data_labels]
            data_as_z_values = np.array(
                [[group_to_z.get(group, np.nan) for group in data_as_groups]]
            )

            if is_vertical:
                data_as_z_values = data_as_z_values.transpose()

            colorscale = [(0.0, default_color)]
            for idx, group in enumerate(all_groups):
                color = group_to_color.get(group, default_color)
                # Imitate discrete scale by using thresholds of size 0.0
                step_low = idx / amount_of_groups
                step_high = (idx + 1) / amount_of_groups
                colorscale.extend([(step_low, color), (step_high, color)])

            trace = go.Heatmap(
                z=data_as_z_values,
                zmin=0,
                zmax=amount_of_groups,
                colorscale=colorscale,
                colorbar=dict(
                    title="Groups",
                    tickvals=[i + 0.5 for i in range(amount_of_groups)],
                    ticktext=all_groups,
                    tickmode="array",
                ),
                hoverinfo="text",
                text=data_labels,
                hovertemplate="Index: %{x}<br>Group: %{text}<extra></extra>",
            )

            return trace

        test_color_map = {
            "CTL": "#ff0000",
            "AD": "#f000f0",
            "Cool Proteins": "#FCE300",
        }

        column_gm_map = create_group_marker_trace(
            self.permuted_column_labels,
            self.column_group_mapping,
            test_color_map,
        )

        _ = fig.add_trace(column_gm_map, row=COL_GM_POS.x, col=COL_GM_POS.y)
        update_xyaxes(fig, COL_GM_POS, visible=False)

        row_gm_map = create_group_marker_trace(
            self.permuted_row_labels,
            self.row_group_mapping,
            test_color_map,
            is_vertical=True,
        )

        _ = fig.add_trace(row_gm_map, row=ROW_GM_POS.x, col=ROW_GM_POS.y)
        update_xyaxes(fig, ROW_GM_POS, visible=False)

        _ = fig.update_layout(plot_bgcolor=plot_bgcolor, showlegend=False)

        fig.show()
        return fig
