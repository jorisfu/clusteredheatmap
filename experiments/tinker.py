import sys, os


sys.path.append("./glustercram")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Random tests
import numpy as np
import pandas as pd

from glustercram.clustergram import ClusteredHeatMap
from glustercram.visu.plotly.builder import PlotlyVisuBuilder
from glustercram.types import Vector

from glustercram.algos import distance, linkage

points: list[Vector] = [
    (1, 1),
    (1.5, 1),
    # (1, 1.5),
    (3.5, 1),
    (3.5, 2),
    # (3.75, 2),
]

example_data = pd.DataFrame(
    {
        "x": [1, 1.5, 3.5, 3.5, 1.01],
        "y": [1, 1, 1, 2, 1.01],
    }
)

significant_proteins = pd.read_csv("./example_data/significant_proteins_df.csv")
significant_proteins = significant_proteins.pivot(
    index="Sample", columns="Protein ID", values="Normalised Ratio H/L"
).T


c = ClusteredHeatMap(
    significant_proteins,
    "nan_euclidean",
    "complete",
    column_group_mappings={
        "Group": {"P10": "CTL", "P11": "CTL", "P3": "AD"},
        "Age": {"P10": "<65", "P11": ">65", "P3": "<65", "P4": ">65"},
    },
    row_group_mappings={
        "Protgroup": {
            "A3KMH1-3": "Cool Proteins",
            "A6NHQ2": "Cool Proteins",
            "Q9Y6R0": "Uncool Proteins",
            "P09382": "Uncool Proteins",
        },
        "Coolness group": {
            "A3KMH1-3": "Yoooo",
            "A6NHQ2": "Noooo",
            "Q9Y6R0": "AAAAAAAAAAAAAAAAAAAAA",
            "P09382": "Yoooo",
        },
    },
    data_column_title="Sample",
    data_row_title="Sample",
)

# fig = c.get_visualization_plotly(
#     heatmap_kwargs={
#         "colorscale": [[0, "#FF0000"], [0.5, "#FFFFFF"], [1, "#0000FF"]],
#         "zmin": -2.5,
#         "zmid": 0,
#         "zmax": 3.5,
#     },
# )
#
# fig.show()

b = PlotlyVisuBuilder(c, vertical_layout="hgd", horizontal_layout="ghd")
b.add_heatmap()
b.add_col_dendrogram()
b.add_row_dendrogram()
b.add_col_group_markers()
b.add_row_group_markers()

b.get_figure().show()
