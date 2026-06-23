# pyright: reportExplicitAny=false

from multiprocessing import current_process
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
import plotly.express as px
import plotly.figure_factory as ff
from plotly import subplots


class Counter:
    def __init__(self) -> None:
        self.value: int = 0

    def next(self) -> int:
        tmp = self.value
        self.value += 1
        return tmp


class Clustergram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance: DistFunName | DistFun,
        linkage: LinkageFunName | LinkageFun,
        column_group_mappings: dict[str, dict[str, str]] | None = None,
        row_group_mappings: dict[str, dict[str, str]] | None = None,
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

        self.column_group_mappings: dict[str, dict[str, str]] = (
            column_group_mappings if column_group_mappings is not None else dict()
        )
        self.row_group_mappings: dict[str, dict[str, str]] = (
            row_group_mappings if row_group_mappings is not None else dict()
        )

    def get_visualization_plotly(
        self,
        *,
        column_title: str = "Column",
        row_title: str = "Row",
        plot_bgcolor: str = "white",
        zaxis_title: str = "Intensity",
        heatmap_nan_color: Color = "#000000",
        heatmap_kwargs: dict[str, Any] | None = None,
        group_marker_colors: list[Color] = px.colors.qualitative.Bold,
    ):
        """
        Returns the computed clustergram as a plotly figure.
        This function is based off the PROTzilla implementation of the Dash Bio clustergram
        but has been heavily refactored and adjusted

        :param zaxis_title: Title for the heatmap legend shown above the color bar
        :param heatmap_nan_color: Color for heatmap cells corresponsing to NaN values
        :param heatmap_kwargs: additional kwargs passed to go.Heatmap
        """

        # Input validation

        # We need at least as many colors as groups
        # TODO: This is wrong, we need to count total no of groups
        if len(group_marker_colors) < len(self.column_group_mappings) + len(
            self.row_group_mappings
        ):
            raise ValueError(f"Too few colors specified in group_marker_colors")

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

        COL_GM_HEIGHT = 2 * len(self.column_group_mappings)
        ROW_GM_HEIGHT = 1 * len(self.row_group_mappings)

        # Layout ratios
        Y_LAYOUT_RATIOS = [30, COL_GM_HEIGHT, 80, 0][::-1]
        X_LAYOUT_RATIOS = [30, ROW_GM_HEIGHT, 80, 0]

        # Colorbar counters
        # Since we're using multiple subplots that each generate colorbars,
        # we need to globally track them to make layouting as dynamic as possible
        colorbar_amount = (
            1 + len(self.row_group_mappings) + len(self.column_group_mappings)
        )
        colorbar_size = 1 / colorbar_amount
        current_colorbar = Counter()

        def current_colorbar_ypos() -> float:
            return current_colorbar.next() / colorbar_amount

        def get_layout_domain_ratios(custom_ratios: list[int]) -> list[list[float]]:
            normalized_ratios: list[float] = [
                i / sum(custom_ratios) for i in custom_ratios
            ]
            domains = []

            start = 0.0
            for boundary in normalized_ratios:
                end = start + boundary
                domains.append([start, end])
                start = end

            return domains

        y_domains = get_layout_domain_ratios(Y_LAYOUT_RATIOS)[::-1]
        x_domains = get_layout_domain_ratios(X_LAYOUT_RATIOS)

        fig = subplots.make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            vertical_spacing=0.0,
            horizontal_spacing=0.0,
        )

        for row in range(rows):
            for col in range(cols):
                if specs[row][col] is not None:
                    _ = fig.update_yaxes(
                        row=row + 1, col=col + 1, domain=y_domains[row]
                    )
                    _ = fig.update_xaxes(
                        row=row + 1, col=col + 1, domain=x_domains[col]
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
        def generate_background_map(
            data: HeatmapMatrix, nan_color: Color
        ) -> go.Heatmap:
            """
            Generates a heatmap with the same dimensions as the primary heatmap but fills
            all cells with the same color. Used to properly visualize NaN values.
            Right now the only way to visualize NaNs in plotly heatmaps, see https://codepen.io/etpinard/pen/xVoJoj
            Also see https://github.com/plotly/plotly.js/issues/7817

            :param data: the primary reference data used for the actual heatmap
            :param nan_color: the color of the background cells (visible for NaN cells in the primary heatmap)

            :return: go.Heatmap object
            """

            background_data = data.copy()
            background_data.fill(0)

            background_map = go.Heatmap(
                z=background_data,
                colorscale=[(0.0, nan_color), (1.0, nan_color)],
                showscale=False,
            )

            return background_map

        if heatmap_nan_color is not None:
            background_map = generate_background_map(
                self.permuted_data, heatmap_nan_color
            )
            _ = fig.add_trace(background_map, HEATMAP_POS.x, HEATMAP_POS.y)

        # TODO: Generalise this and cast all groupings
        col_label_matrix = np.broadcast_to(
            self.permuted_column_labels, self.permuted_data.shape
        )
        row_label_matrix = np.broadcast_to(
            np.array(self.permuted_row_labels)[:, None], self.permuted_data.shape
        )

        # custom_data[:, :, 0] will be columns, custom_data[:, :, 1] will be rows
        custom_data = np.dstack((col_label_matrix, row_label_matrix))

        heatmap = go.Heatmap(
            z=self.permuted_data,
            colorbar=dict(
                title=zaxis_title,
                yanchor="bottom",
                y=current_colorbar_ypos(),
                len=colorbar_size,
                # tickmode="array",
                # tickvals=(zmin, target_data_midpoint, zmax),
            ),
            customdata=custom_data,
            hovertemplate=f"{column_title}: %{{customdata[0]}}<br>{row_title}: %{{customdata[1]}}<br>{zaxis_title}: %{{z}} <extra></extra>",
            **heatmap_kwargs,
        )

        _ = fig.add_trace(
            heatmap,
            row=HEATMAP_POS.x,
            col=HEATMAP_POS.y,
        )

        update_xyaxes(fig, HEATMAP_POS, showticklabels=False)
        _ = fig.update_xaxes(row=HEATMAP_POS.x, col=HEATMAP_POS.y, range=[-0.5, len(self.permuted_column_labels) - 0.5])
        _ = fig.update_yaxes(row=HEATMAP_POS.x, col=HEATMAP_POS.y, range=[-0.5, len(self.permuted_row_labels) - 0.5])

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

        # Downscale to match heatmap axis
        for trace in cols_dendro_traces:
            trace["x"] = np.array(trace["x"] - 5) / 10

        add_tracelist(fig, COL_DENDRO_POS, cols_dendro_traces)
        update_xyaxes(fig, COL_DENDRO_POS, **dendro_axes_layout)

        # Rows
        rows_dendro_traces = ff._dendrogram._Dendrogram(
            self.data_rows,
            orientation="right",
            distfun=lambda _: None,
            linkagefun=lambda _: self.linkage_matrix_rows,  # Always use precomputed matrix
        ).data

        # Downscale to match heatmap axis
        for trace in rows_dendro_traces:
            trace["y"] = np.array(trace["y"] - 5) / 10

        add_tracelist(fig, ROW_DENDRO_POS, rows_dendro_traces)
        update_xyaxes(fig, ROW_DENDRO_POS, **dendro_axes_layout)

        ## GROUP MARKERS
        gm_colorgen = iter(group_marker_colors)

        def create_group_marker_trace(
            data_labels: list[str],
            label_to_group: dict[str, str],
            default_color: str = "#000000",
            is_vertical: bool = False,
            colorbar_ypos: float = 0.0,
            colorbar_size: float = 1.0,
            legend_title: str = "Group",
            axis_title: str = "Axis value",
            group_no: int = 0,
            groups_on_axis: int = 1,
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

            data_as_z_values = np.full([groups_on_axis, len(data_labels)], np.nan)
            data_as_z_values[group_no] = np.array(
                [[group_to_z.get(group, np.nan) for group in data_as_groups]]
            )

            group_label_matrix = [[""] * len(data_labels)] * groups_on_axis
            group_label_matrix[group_no] = [l or "" for l in data_as_groups]
            group_label_matrix = np.array(group_label_matrix)

            data_label_matrix = [[""] * len(data_labels)] * groups_on_axis
            data_label_matrix[group_no] = data_labels
            data_label_matrix = np.array(data_label_matrix)

            if is_vertical:
                data_as_z_values = data_as_z_values.transpose()
                group_label_matrix = group_label_matrix.transpose()
                data_label_matrix = data_label_matrix.transpose()

            custom_data = np.dstack((data_label_matrix, group_label_matrix))

            colorscale = [(0.0, default_color)]
            for idx, group in enumerate(all_groups):
                color = next(gm_colorgen)
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
                    title=legend_title,
                    tickvals=[i + 0.5 for i in range(amount_of_groups)],
                    ticktext=all_groups,
                    tickmode="array",
                    y=colorbar_ypos,
                    yanchor="bottom",
                    len=colorbar_size,
                ),
                customdata=custom_data,
                text=data_labels,
                hovertemplate=f"{axis_title} %{{customdata[0]}}<br>{legend_title}: %{{customdata[1]}}<br><extra></extra>",
                hoverongaps=False,
            )

            return trace

        for idx, (label, mapping) in enumerate(self.column_group_mappings.items()):
            column_gm_map = create_group_marker_trace(
                self.permuted_column_labels,
                mapping,
                colorbar_size=colorbar_size,
                colorbar_ypos=current_colorbar_ypos(),
                legend_title=label,
                group_no=idx,
                groups_on_axis=len(self.column_group_mappings),
                axis_title=column_title,
            )
            _ = fig.add_trace(column_gm_map, row=COL_GM_POS.x, col=COL_GM_POS.y)

        update_xyaxes(fig, COL_GM_POS, visible=False)

        for idx, (label, mapping) in enumerate(self.row_group_mappings.items()):
            row_gm_map = create_group_marker_trace(
                self.permuted_row_labels,
                mapping,
                is_vertical=True,
                colorbar_size=colorbar_size,
                colorbar_ypos=current_colorbar_ypos(),
                legend_title=label,
                group_no=idx,
                groups_on_axis=len(self.row_group_mappings),
                axis_title=row_title,
            )

            _ = fig.add_trace(row_gm_map, row=ROW_GM_POS.x, col=ROW_GM_POS.y)

        update_xyaxes(fig, ROW_GM_POS, visible=False)

        _ = fig.update_layout(plot_bgcolor=plot_bgcolor, showlegend=False)

        ## AXIS SYNCING
        heatmap_x_id = next(
            fig.select_xaxes(row=HEATMAP_POS.x, col=HEATMAP_POS.y)
        ).plotly_name.replace("axis", "")
        heatmap_y_id = next(
            fig.select_yaxes(row=HEATMAP_POS.x, col=HEATMAP_POS.y)
        ).plotly_name.replace("axis", "")

        _ = fig.update_xaxes(matches=heatmap_x_id, row=COL_GM_POS.x, col=COL_GM_POS.y)
        _ = fig.update_yaxes(matches=heatmap_y_id, row=ROW_GM_POS.x, col=ROW_GM_POS.y)
        _ = fig.update_xaxes(matches=heatmap_x_id, row=COL_DENDRO_POS.x, col=COL_DENDRO_POS.y)
        _ = fig.update_yaxes(matches=heatmap_y_id, row=ROW_DENDRO_POS.x, col=ROW_DENDRO_POS.y)

        return fig
