import sys, os

sys.path.append("./glustercram")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Random tests
import numpy as np
import pandas as pd

from glustercram.clustergram import Clustergram
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


c = Clustergram(
    significant_proteins,
    "nan_euclidean",
    "complete",
)

_ = c.get_visualization_plotly(
    heatmap_kwargs={
        "colorscale": [[0, "#FF0000"], [0.5, "#FFFFFF"], [1, "#0000FF"]],
        "zmin": -2.5,
        "zmid": 0,
        "zmax": 3.5,
    }
)
