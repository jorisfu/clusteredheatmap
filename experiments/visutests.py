import sys, os

sys.path.append("./glustercram")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import plotly.graph_objects as go
import numpy as np

from glustercram.types import HeatmapMatrix
from glustercram.visu.heatmap import heatmap

d1: HeatmapMatrix = np.array([[1, 2, 3], [3, 2, 2], [3, np.nan, 3]])

fig = go.Figure(data=heatmap(d1, nan_color="#808080", hovertemplate="Asdjasd"))

fig.show()
