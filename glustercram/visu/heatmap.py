import plotly.graph_objects as go

from glustercram.types import Color, HeatmapMatrix


def generate_background_map(data: HeatmapMatrix, nan_color: Color) -> go.Heatmap:
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


def heatmap(
    data: HeatmapMatrix,
    *,
    nan_color: Color | None = None,
    heatmap_legend_title: str = "",
    **kwargs
):
    """
    Wrapper for go.Heatmap, creates a plotly heatmap

    :param data: the data to plot
    :param nan_color: the color of the cells for NaN values. Defaults to transparent if None
    """
    fig = go.Figure()

    heatmap = go.Heatmap(
        z=data,
        colorbar=dict(
            title=heatmap_legend_title,
            yanchor="bottom",
            y=0.0,
            # tickmode="array",
            # tickvals=(zmin, target_data_midpoint, zmax),
        ),
        **kwargs  # pyright: ignore[reportUnknownArgumentType]
    )

    if nan_color is not None:
        background_map = generate_background_map(data, nan_color)
        _ = fig.add_trace(background_map)

    _ = fig.add_trace(heatmap)

    # TODO: Fix plotly upstream or find a way to add a whole figure as a subplot
    return [background_map, heatmap]
