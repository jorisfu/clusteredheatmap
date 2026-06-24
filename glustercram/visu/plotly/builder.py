from ast import Sub
from enum import StrEnum
from typing import Any, Generator


import numpy as np

from plotly.basedatatypes import BaseTraceType
import plotly.express as px
from plotly import subplots
from plotly.graph_objs import Figure
from glustercram.clustergram import ClusteredHeatMap
from glustercram.types import Color, HeatmapMatrix, LayoutPoint


class LayoutError(Exception):
    pass


class SubplotType(StrEnum):
    COL_DENDRO = "col_dendro"
    ROW_DENDRO = "row_dendro"
    COL_GROUPMARKERS = "col_groupmarkers"
    ROW_GROUPMARKERS = "row_groupmarkers"
    HEATMAP = "heatmap"


class PlotlyVisuBuilder:


    def _validate_layout_string(self, s: str) -> None:
        if len(s) == 0:
            raise LayoutError(
                "Empty layout string given, must include at least heatmap"
            )

        if len(s) > 3:
            raise LayoutError("Layout string is too long")

        if len(s) != len(set(s)):
            raise LayoutError(
                "Layout string may only specify one position for each subplot"
            )

        if set(s) - {"d", "g", "h"}:
            raise LayoutError("Layout string has illegal characters, may only be d/g/h")

        if "h" not in s:
            raise LayoutError("Layout string does not specify heatmap position")

    def _build_layout(
        self, horizontal_layout: str, vertical_layout: str
    ) -> dict[SubplotType, LayoutPoint]:
        subplot_positions: dict[SubplotType, LayoutPoint] = dict()

        heatmap_position = LayoutPoint(
            vertical_layout.index("h") + 1,
            horizontal_layout.index("h") + 1,
        )

        for row, content in enumerate(vertical_layout):
            match content:
                case "d":
                    subplot_positions[SubplotType.COL_DENDRO] = LayoutPoint(
                        row + 1, heatmap_position.col
                    )
                case "g":
                    subplot_positions[SubplotType.COL_GROUPMARKERS] = LayoutPoint(
                        row + 1, heatmap_position.col
                    )
                case _:
                    pass

        for col, content in enumerate(horizontal_layout):
            match content:
                case "d":
                    subplot_positions[SubplotType.ROW_DENDRO] = LayoutPoint(
                        heatmap_position.row, col + 1
                    )
                case "g":
                    subplot_positions[SubplotType.ROW_GROUPMARKERS] = LayoutPoint(
                        heatmap_position.row, col + 1
                    )
                case _:
                    pass

        return subplot_positions

    def _build_figure(self) -> Figure:
        specs: list[list[None | dict[str, Any]]] = [
            [None for _ in range(self.subplot_cols)] for _ in range(self.subplot_rows)
        ]

        for subplot, pos in self.subplot_positions.items():
            specs[pos.row - 1][pos.col - 1] = dict()

        return subplots.make_subplots(
            rows=self.subplot_rows,
            cols=self.subplot_cols,
            specs=specs,
            vertical_spacing=0.0,
            horizontal_spacing=0.0,
        )

    # TODO: Parametrize
    def _default_distinct_colorgen(self) -> Generator[Color]:
        return (y for y in px.colors.qualitative.Bold)


    def __init__(
        self,
        chm: ClusteredHeatMap,
        vertical_layout: str = "",
        horizontal_layout: str = "",
    ) -> None:
        """
        Builder for a plotly-based visualization of a clustered heatmap.
        Needs some basic layout info as a starting point, subsequent
        calls then build the visualization step-by-step.

        :param chm: the ClusteredHeatMap object to visualize.
        :param vertical_layout: Layout of elements on the vertical axis of the plot.
            Elements include
            'd' for dendrogram
            'g' for group markers
            'h' for heatmap
            Example: "dgh" for dendrogram on the left, group markers in the middle and
            heatmap on the right or "hd" for dendrogram on the right with no group
            markers.
        :param horizontal_layout: Layout of elements on the horizontal axis of the plot.
            See vertical_layout for more info.
        """

        self.chm: ClusteredHeatMap = chm
        self.helpers: PlotlyHelpers = PlotlyHelpers(self)

        self._validate_layout_string(vertical_layout)
        self._validate_layout_string(horizontal_layout)

        self.subplot_rows: int = len(vertical_layout) or 1
        self.subplot_cols: int = len(horizontal_layout) or 1

        # Subplot positions are 1-based for plotly references
        self.subplot_positions: dict[SubplotType, LayoutPoint] = self._build_layout(
            horizontal_layout, vertical_layout
        )

        self.fig: Figure = self._build_figure()

        # Used for group markers if no custom colors are given
        self.distinct_colorgen: Generator[Color] = self._default_distinct_colorgen()


    def add_heatmap(
        self,
        nan_color: Color = "#000000"
    ) -> None:
        def generate_background_map(
            data: HeatmapMatrix, nan_color: Color
        ) -> BaseTraceType:
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

        pos = self.subplot_positions[SubplotType.HEATMAP]

        # TODO: Check here if we have nans in data
        if True:
            background_map = generate_background_map(self.chm.permuted_data, nan_color)
            self.helpers.add_trace(background_map, pos)

        # TODO: Generalise this and cast all groupings
        # So we can get group info in tooltip?
        col_label_matrix = np.broadcast_to(
            self.chm.permuted_column_labels, self.chm.permuted_data.shape
        )
        row_label_matrix = np.broadcast_to(
            np.array(self.chm.permuted_row_labels)[:, None], self.chm.permuted_data.shape
        )

        # custom_data[:, :, 0] will be columns, custom_data[:, :, 1] will be rows
        custom_data = np.dstack((col_label_matrix, row_label_matrix))

        heatmap = go.Heatmap(
            z=self.chm.permuted_data,
            colorbar=dict(
                title=self.chm.data_z_title,
                yanchor="bottom",
                y=current_colorbar_ypos(),
                len=colorbar_size,
                # tickmode="array",
                # tickvals=(zmin, target_data_midpoint, zmax),
            ),
            customdata=custom_data,
            hovertemplate=f"{self.data_column_title}: %{{customdata[0]}}<br>{self.data_row_title}: %{{customdata[1]}}<br>{self.data_z_title}: %{{z}} <extra></extra>",
            **heatmap_kwargs,
        )

        _ = fig.add_trace(
            heatmap,
            row=HEATMAP_POS.row,
            col=HEATMAP_POS.col,
        )
        
class PlotlyHelpers:
    def __init__(self, builder: PlotlyVisuBuilder) -> None:
        self._builder: PlotlyVisuBuilder = builder

    def add_trace(self, trace: BaseTraceType, pos: LayoutPoint) -> None:
        _ = self._builder.fig.add_trace(trace, pos.row, pos.col)
