import sys, os


sys.path.append("./clusteredheatmap")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Random tests
import numpy as np
import pandas as pd

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder
from clusteredheatmap.types import Vector

from clusteredheatmap.algos import distance, linkage

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
    distance="dixon_pds_euclidean",
    linkage="complete",
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
    data_row_title="Protein",
)

b = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh")
b.add_heatmap(_zmin=-2.5, _zmid="median", _zmax=3.5, ticktext_prefix=("start: ", "mid: ", "end: "))
b.add_col_dendrogram()
b.add_row_dendrogram()
b.add_col_group_markers(_color_overrides={"Age": {"<65": "#FCE300", ">65": "#ABD310"}})
b.add_row_group_markers(
    _color_overrides={
        "Protgroup": {"Cool Proteins": "#30ff65", "Uncool Proteins": "#ff1234"}
    }
)
b.add_col_ticks(anchor_subplot="h", side="bottom")
# b.add_row_ticks(anchor_subplot="g", side="left")
b.get_figure().show()

# fig = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh").autobuild()
# fig.show()
