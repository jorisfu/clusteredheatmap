# pyright: reportExplicitAny=false
from enum import StrEnum
from typing import Any, Generator, Literal


import numpy as np

import plotly.colors
from plotly.basedatatypes import BaseTraceType
import plotly.express as px
from plotly import subplots
from plotly.graph_objs import Figure
from glustercram.clustergram import ClusteredHeatMap
from glustercram.types import Color, HeatmapMatrix, LayoutPoint
import plotly.graph_objects as go
import plotly.figure_factory as ff


class LayoutError(Exception):
    pass


class ColorError(Exception):
    pass


class SubplotType(StrEnum):
    COL_DENDRO = "col_dendro"
    ROW_DENDRO = "row_dendro"
    COL_GROUPMARKERS = "col_groupmarkers"
    ROW_GROUPMARKERS = "row_groupmarkers"
    HEATMAP = "heatmap"


DENDRO_AXES_LAYOUT = {
    "showline": False,
    "showgrid": False,
    "showticklabels": False,
}

DEFAULT_HEATMAP_COLORSCALE = [(0.0, "#FF0000"), (0.5, "#FFFFFF"), (1.0, "#0000FF")]


class PlotlyVisuBuilder:

    def __init__(
        self,
        chm: ClusteredHeatMap,
        *,
        vertical_layout: str = "dgh",
        horizontal_layout: str = "dgh",
        background_color: Color = "white",
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

        self.vertical_layout: str = vertical_layout
        self.horizontal_layout: str = horizontal_layout

        self.subplot_rows: int = len(vertical_layout) or 1
        self.subplot_cols: int = len(horizontal_layout) or 1

        # Subplot positions are 1-based for plotly references
        self.subplot_positions: dict[SubplotType, LayoutPoint] = self._build_layout(
            horizontal_layout, vertical_layout
        )

        self.fig: Figure = self._build_figure()
        _ = self.fig.update_layout(plot_bgcolor=background_color, showlegend=False)

        # Need to save these for axes synchronization
        self._heatmap_x_id = next(
            self.fig.select_xaxes(
                **self.subplot_positions[SubplotType.HEATMAP]._asdict()
            )
        ).plotly_name.replace("axis", "")
        self._heatmap_y_id = next(
            self.fig.select_yaxes(
                **self.subplot_positions[SubplotType.HEATMAP]._asdict()
            )
        ).plotly_name.replace("axis", "")

        # Used for group markers if no custom colors are given
        self.distinct_colorgen: Generator[Color] = self._default_distinct_colorgen()

        # Required for size/pos of colorbars
        self._colorbar_amount: int = (
            1 + len(self.chm.row_group_mappings) + len(self.chm.column_group_mappings)
        )
        self._colorbar_position: Generator[float, None, None] = (
            self._colorbar_position_generator(self._colorbar_amount)
        )

        self._relative_subplot_widths: dict[SubplotType, int] = dict()
        self._relative_subplot_heights: dict[SubplotType, int] = dict()

    def autobuild(self) -> Figure:
        """
        Automatically builds a clustered heatmap visualization using sane defaults.
        """
        self.add_heatmap()

        for char in self.vertical_layout:
            match char:
                case "d":
                    self.add_col_dendrogram()
                case "g":
                    self.add_col_group_markers()
                case _:
                    pass

        for char in self.horizontal_layout:
            match char:
                case "d":
                    self.add_row_dendrogram()
                case "g":
                    self.add_row_group_markers()
                case _:
                    pass

        return self.get_figure()

    def get_figure(self) -> Figure:
        # For performance reasons we only do this right before returning the fully constructed figure
        self._apply_display_ratios()
        return self.fig

    ##
    ## LAYOUT AND DISPLAY
    ##

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

        subplot_positions[SubplotType.HEATMAP] = heatmap_position

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

    def _sync_to_heatmap(self, axis: Literal["x", "y"], pos: LayoutPoint):
        match axis:
            case "x":
                _ = self.fig.update_xaxes(matches=self._heatmap_x_id, **pos._asdict())
            case "y":
                _ = self.fig.update_yaxes(matches=self._heatmap_y_id, **pos._asdict())

    def _apply_display_ratios(self):
        y_layout_ratios: list[int] = []
        x_layout_ratios: list[int] = []

        for char in self.vertical_layout:
            match char:
                case "d":
                    y_layout_ratios.append(
                        self._relative_subplot_heights.get(SubplotType.COL_DENDRO, 0)
                    )
                case "g":
                    y_layout_ratios.append(
                        self._relative_subplot_heights.get(
                            SubplotType.COL_GROUPMARKERS, 0
                        )
                    )
                case "h":
                    y_layout_ratios.append(
                        self._relative_subplot_heights.get(SubplotType.HEATMAP, 0)
                    )
                case _:
                    pass

        for char in self.horizontal_layout:
            match char:
                case "d":
                    x_layout_ratios.append(
                        self._relative_subplot_widths.get(SubplotType.ROW_DENDRO, 0)
                    )
                case "g":
                    x_layout_ratios.append(
                        self._relative_subplot_widths.get(
                            SubplotType.ROW_GROUPMARKERS, 0
                        )
                    )
                case "h":
                    x_layout_ratios.append(
                        self._relative_subplot_widths.get(SubplotType.HEATMAP, 0)
                    )
                case _:
                    pass

        y_layout_ratios = y_layout_ratios[::-1]

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

        y_domains = get_layout_domain_ratios(y_layout_ratios)[::-1]
        x_domains = get_layout_domain_ratios(x_layout_ratios)

        for row in range(self.subplot_rows):
            for col in range(self.subplot_cols):
                _ = self.fig.update_yaxes(
                    row=row + 1, col=col + 1, domain=y_domains[row]
                )
                _ = self.fig.update_xaxes(
                    row=row + 1, col=col + 1, domain=x_domains[col]
                )

    ##
    ## COLORBARS AND DISTINCT COLOR SEQUENCE GENERATION
    ##

    # TODO: Parametrize
    def _default_distinct_colorgen(self) -> Generator[Color, None, None]:
        return (y for y in px.colors.qualitative.Bold)

    def _new_colorbar(self, **kwargs: Any):
        """
        Returns a dict specifying a new colorbar legend.
        Position, size and anchor get handled by this,
        all other parameters can be freely passed
        """
        return dict(
            yanchor="bottom",
            x=1.0,
            y=next(self._colorbar_position),
            len=1 / self._colorbar_amount,
            **kwargs,
        )

    def _colorbar_position_generator(self, total_amount: int):
        for i in range(total_amount):
            yield i / total_amount

    ##
    ## HEATMAP AND CONTINUOUS COLORSCALES
    ##

    def _build_asymmetric_colorscale(
        self,
        _colorscale: str | list[tuple[float, Color]],
        zmin: float,
        zmax: float,
        zmid: float,
    ):
        """
        Adjusts a divergent colorscale for a heatmap.
        The colors are redistributed to allow for asymmetry according
        to the data thresholds given.
        Note that just passing zmid to plotly's heatmap does not work like this, hence
        this implementation

        :param _colorscale: Either the colorscale to adjust or the name of the plotly
            colorscale to use.
        :param zmin: the data value corresponding to the minimum value (0.0) of the colorscale
        :param zmid: the data value corresponding to the center of the colorscale.
            The resulting colorscale's center point will be adjusted to this.
        :param zmax: the data value corresponding to the maximum value (1.0) of the colorscale
        """

        colorscale: list[list[float | Color]] | list[tuple[float, Color]] = []
        if isinstance(_colorscale, str):
            colorscale = plotly.colors.get_colorscale(_colorscale)
        else:
            colorscale = _colorscale

        if 0.5 not in [c[0] for c in colorscale]:
            raise ColorError(
                "Colorscale for heatmap is missing a clearly defined midpoint (color at 0.5)"
            )

        color_midpoint = (zmid - zmin) / (zmax - zmin)

        new_colorscale = []
        for tick in filter(lambda tick: tick[0] <= 0.5, colorscale):
            adjusted_tick_base = lerp(0, color_midpoint, tick[0] / 0.5)
            new_colorscale.append((adjusted_tick_base, tick[1]))

        for tick in filter(lambda tick: tick[0] > 0.5, colorscale):
            adjusted_tick_base = lerp(
                color_midpoint, 1, (tick[0] - color_midpoint) / (1 - color_midpoint)
            )
            new_colorscale.append((adjusted_tick_base, tick[1]))

        return new_colorscale

    def add_heatmap(
        self,
        *,
        relative_width: int = 80,
        relative_height: int = 80,
        nan_color: Color = "#000000",
        colorscale: str | list[tuple[float, Color]] | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zmid: float | None = None,
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

        target_position = self.subplot_positions[SubplotType.HEATMAP]

        ## Colorscale calculation
        if zmin is None:
            zmin = float(np.nanmin(self.chm.permuted_data))

        if zmax is None:
            zmax = float(np.nanmax(self.chm.permuted_data))

        if zmid is None:
            zmid = np.average([zmin, zmax])

        if not (zmin <= zmid <= zmax):
            raise ValueError(
                f"Heatmap zmin ({str(zmin)}), zmid ({str(zmid)}) and zmax ({str(zmax)}) must be in increasing order."
            )

        if colorscale is None:
            colorscale = DEFAULT_HEATMAP_COLORSCALE

        colorscale = self._build_asymmetric_colorscale(colorscale, zmin, zmax, zmid)

        ## Background map for NaNs
        # TODO: Check here if we have nans in data
        if True:
            background_map = generate_background_map(self.chm.permuted_data, nan_color)
            self.helpers.add_trace(background_map, target_position)

        ## Custom data for tooltip
        # TODO: Generalise this and cast all groupings
        # So we can get group info in tooltip?
        col_label_matrix = np.broadcast_to(
            self.chm.permuted_column_labels, self.chm.permuted_data.shape
        )
        row_label_matrix = np.broadcast_to(
            np.array(self.chm.permuted_row_labels)[:, None],
            self.chm.permuted_data.shape,
        )

        # custom_data[:, :, 0] will be columns, custom_data[:, :, 1] will be rows
        custom_data = np.dstack((col_label_matrix, row_label_matrix))

        heatmap = go.Heatmap(
            z=self.chm.permuted_data,
            colorbar=self._new_colorbar(
                title=self.chm.data_z_title,
                tickmode="array",
                tickvals=(zmin, zmid, zmax),
            ),
            colorscale=colorscale,
            customdata=custom_data,
            hovertemplate=f"{self.chm.data_column_title}: %{{customdata[0]}}<br>{self.chm.data_row_title}: %{{customdata[1]}}<br>{self.chm.data_z_title}: %{{z}} <extra></extra>",
            zmin=zmin,
            zmax=zmax,
        )

        self.helpers.add_trace(heatmap, target_position)
        self.helpers.update_xyaxes(target_position, showticklabels=False)

        # Required to align with dendrograms
        _ = self.fig.update_xaxes(
            range=[-0.5, len(self.chm.permuted_column_labels) - 0.5],
            **target_position._asdict(),
        )
        _ = self.fig.update_yaxes(
            range=[-0.5, len(self.chm.permuted_row_labels) - 0.5],
            **target_position._asdict(),
        )

        self._relative_subplot_widths[SubplotType.HEATMAP] = relative_width
        self._relative_subplot_heights[SubplotType.HEATMAP] = relative_height

    ##
    ## DENDROGRAMS
    ##

    def add_col_dendrogram(self, relative_height: int = 30):
        if not self.chm.cluster_columns:
            raise ValueError(
                "Columns were not clustered, no dendrogram can be plotted."
            )

        target_position = self.subplot_positions[SubplotType.COL_DENDRO]

        if "d" not in self.vertical_layout:
            raise LayoutError(
                "Cannot add column dendrogram as position is not specified in layout"
            )
        orientation = (
            "bottom"
            if self.vertical_layout.index("d") < self.vertical_layout.index("h")
            else "top"
        )

        cols_dendro_traces = ff._dendrogram._Dendrogram(
            self.chm.data_cols,
            orientation=orientation,
            distfun=lambda _: None,
            linkagefun=lambda _: self.chm.linkage_matrix_cols,  # Always use precomputed matrix
        ).data

        # Downscale to match heatmap axis
        invert_scale = (
            -1 if orientation == "top" else 1
        )  # Sadly the ff dendrogram gets mirrored so we need this
        for trace in cols_dendro_traces:
            trace["x"] = np.array(trace["x"] - 5 * invert_scale) * invert_scale / 10

        self.helpers.add_tracelist(target_position, cols_dendro_traces)
        self.helpers.update_xyaxes(target_position, **DENDRO_AXES_LAYOUT)
        self._sync_to_heatmap("x", target_position)
        self._relative_subplot_heights[SubplotType.COL_DENDRO] = relative_height

    def add_row_dendrogram(self, relative_width: int = 30):
        if not self.chm.cluster_rows:
            raise ValueError("Rows were not clustered, no dendrogram can be plotted.")

        target_position = self.subplot_positions[SubplotType.ROW_DENDRO]

        if "d" not in self.horizontal_layout:
            raise LayoutError(
                "Cannot add row dendrogram as position is not specified in layout"
            )
        orientation = (
            "right"
            if self.horizontal_layout.index("d") < self.horizontal_layout.index("h")
            else "left"
        )

        rows_dendro_traces = ff._dendrogram._Dendrogram(
            self.chm.data_rows,
            orientation=orientation,
            distfun=lambda _: None,
            linkagefun=lambda _: self.chm.linkage_matrix_rows,  # Always use precomputed matrix
        ).data

        # Downscale to match heatmap axis
        invert_scale = (
            -1 if orientation == "left" else 1
        )  # Sadly the ff dendrogram gets mirrored so we need this
        for trace in rows_dendro_traces:
            trace["y"] = np.array(trace["y"] - 5 * invert_scale) * invert_scale / 10

        self.helpers.add_tracelist(target_position, rows_dendro_traces)
        self.helpers.update_xyaxes(target_position, **DENDRO_AXES_LAYOUT)
        self._sync_to_heatmap("y", target_position)
        self._relative_subplot_widths[SubplotType.ROW_DENDRO] = relative_width

    ##
    ## GROUP MARKERS
    ##

    def _create_group_marker_trace(
        self,
        data_labels: list[str],
        label_to_group: dict[str, str],
        *,
        group_to_color: dict[str, Color] | None = None,
        legend_title: str = "Group",
        axis_title: str = "Axis value",
        groups_on_axis: int = 1,
        group_no: int = 0,
        is_vertical: bool = False,
    ):
        """
        Creates a trace used for group markers.

        :param data_labels: The labels of the data as ordered on the axis to mark
        :param label_to_group: Mapping of data labels to group identifiers.
            If incomplete, mapping is performed to None and no color marker will be set
        :param group_to_color: Mapping of group identifiers to colors (optional).
            If given, must fully map all groups to colors.
            By default, a distinct color generator is used and no mapping needs to
            be provided.
        :param groups_on_axis: If multiple group markers are used, this is the total number
            of group markers that are plotted on the same axis.
        :param group_no: If multiple group markers are used, this is the position
            of the current group marker within the group marker stack
        :param is_vertical: True iff the group markers are to be arranged vertically
        """

        all_groups = sorted(list(set(label_to_group.values())))
        amount_of_groups = len(all_groups)

        if group_to_color is not None and amount_of_groups > len(group_to_color):
            raise ColorError("Provided color mapping does not map all groups to colors")

        # Embedding like this required for continuous colorscale
        group_to_z: dict[str | None, float] = {
            g: i + 0.5 for i, g in enumerate(all_groups)  # Center within bin of size 1
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

        colorscale: list[tuple[float, Color]] = []
        for idx, group in enumerate(all_groups):
            if group_to_color is not None:
                color = group_to_color[group]
            else:
                color = next(self.distinct_colorgen)
            # Imitate discrete scale by using thresholds of size 0.0
            step_low = idx / amount_of_groups
            step_high = (idx + 1) / amount_of_groups
            colorscale.extend([(step_low, color), (step_high, color)])

        trace = go.Heatmap(
            z=data_as_z_values,
            zmin=0,
            zmax=amount_of_groups,
            colorscale=colorscale,
            colorbar=self._new_colorbar(
                title=legend_title,
                tickvals=[i + 0.5 for i in range(amount_of_groups)],
                ticktext=all_groups,
                tickmode="array",
            ),
            customdata=custom_data,
            text=data_labels,
            hovertemplate=f"{axis_title} %{{customdata[0]}}<br>{legend_title}: %{{customdata[1]}}<br><extra></extra>",
            hoverongaps=False,
        )

        return trace

    def add_col_group_markers(
        self,
        relative_height_per_marker: int = 2,
        _color_overrides: dict[str, dict[str, Color]] | None = None,
    ):
        """
        Adds group markers per column.

        :param relative_height_per_marker: visual height of each group marker (relative to entire plot)
        :param _color_overrides: custom colors for groups. Mapping is performed from the name of the mapping
            (see column_group_mappings in ClusteredHeatMap) to a dict mapping each group to a specific color.
            Color mapping must be complete for all groups to override.
            If no mapping is specified, the default color generator gets used.
        """
        target_position: LayoutPoint = self.subplot_positions[
            SubplotType.COL_GROUPMARKERS
        ]
        color_overrides = _color_overrides or dict()

        for idx, (label, mapping) in enumerate(self.chm.column_group_mappings.items()):
            column_gm_map = self._create_group_marker_trace(
                self.chm.permuted_column_labels,
                mapping,
                legend_title=label,
                group_no=idx,
                groups_on_axis=len(self.chm.column_group_mappings),
                axis_title=self.chm.data_column_title,
                group_to_color=color_overrides.get(label),
            )
            self.helpers.add_trace(column_gm_map, target_position)

        self.helpers.update_xyaxes(target_position, visible=False)
        self._sync_to_heatmap("x", target_position)
        self._relative_subplot_heights[SubplotType.COL_GROUPMARKERS] = (
            relative_height_per_marker * len(self.chm.column_group_mappings)
        )

    def add_row_group_markers(
        self,
        relative_width_per_marker: int = 1,
        _color_overrides: dict[str, dict[str, Color]] | None = None,
    ):
        """
        Adds group markers per column.

        :param relative_width_per_marker: visual width of each group marker (relative to entire plot)
        :param _color_overrides: custom colors for groups. Mapping is performed from the name of the mapping
            (see row_group_mappings in ClusteredHeatMap) to a dict mapping each group to a specific color.
            Color mapping must be complete for all groups to override.
            If no mapping is specified, the default color generator gets used.
        """
        target_position: LayoutPoint = self.subplot_positions[
            SubplotType.ROW_GROUPMARKERS
        ]
        color_overrides = _color_overrides or dict()

        for idx, (label, mapping) in enumerate(self.chm.row_group_mappings.items()):
            row_gm_map = self._create_group_marker_trace(
                self.chm.permuted_row_labels,
                mapping,
                legend_title=label,
                group_no=idx,
                groups_on_axis=len(self.chm.row_group_mappings),
                axis_title=self.chm.data_row_title,
                is_vertical=True,
                group_to_color=color_overrides.get(label),
            )
            self.helpers.add_trace(row_gm_map, target_position)

        self.helpers.update_xyaxes(target_position, visible=False)
        self._sync_to_heatmap("y", target_position)
        self._relative_subplot_widths[SubplotType.ROW_GROUPMARKERS] = (
            relative_width_per_marker * len(self.chm.row_group_mappings)
        )


class PlotlyHelpers:
    def __init__(self, builder: PlotlyVisuBuilder) -> None:
        self._builder: PlotlyVisuBuilder = builder

    def add_trace(self, trace: BaseTraceType, pos: LayoutPoint) -> None:
        _ = self._builder.fig.add_trace(trace, pos.row, pos.col)

    def add_tracelist(self, pos: LayoutPoint, traces: list[BaseTraceType]):
        """
        Adds multiple traces to the same subplot
        """
        _ = self._builder.fig.add_traces(
            traces,
            rows=[pos.row] * len(traces),
            cols=[pos.col] * len(traces),
        )

    def update_xyaxes(self, pos: LayoutPoint, **kwargs: Any):
        """
        Updates x and y axis layout args for a subplot
        """
        _ = self._builder.fig.update_xaxes(row=pos.row, col=pos.col, **kwargs)
        _ = self._builder.fig.update_yaxes(row=pos.row, col=pos.col, **kwargs)


def lerp(start: float, end: float, alpha: float) -> float:
    """
    Performs linear interpolation

    :param start: first interpolation value
    :param end: second interpolation value
    :param alpha: interpolation factor (between 0 and 1)
    """
    return (1 - alpha) * start + alpha * end
