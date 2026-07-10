import sys, os


sys.path.append("./clusteredheatmap")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Random tests
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from clusteredheatmap.chm import ClusteredHeatMap
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder
from clusteredheatmap.types import Vector

SIZE_N = 5000

def remove_negative_values_from_correlation_matrix(correlation_matrix):#todo: rename function
    """removes all negative values, so that all values are in [0,1] and converts correlation c into distance d with d = 1-c"""
    distance_matrix = correlation_matrix.to_numpy()
    #distance_matrix = np.where(distance_matrix >= 0, distance_matrix, 0)
    #distance_matrix = 1 - distance_matrix
    distance_matrix = np.clip(distance_matrix, -0.999999, 0.999999) #war notw. für den validity score von hdbscan ->wenn irgendwo eine 1 drin steht, wird das für die dist-matrix zu 0 und dann teilen wir im Algo durch 0
    distance_matrix = np.sqrt(2*(1-distance_matrix))
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

corr = pd.read_csv("./example_data/largestuff/correlation.csv")
corr = pd.DataFrame(corr.to_numpy()[:SIZE_N, :SIZE_N])


dist = squareform(remove_negative_values_from_correlation_matrix(corr))


# Pre-compute linkage to reduce redundancy
z = linkage(dist, "average")

c = ClusteredHeatMap(
    corr,
    distance="euclidean",
    linkage="average",
    precomputed_dist_cols=dist,
    precomputed_dist_rows=dist,
    precomputed_linkage_cols=z,
    precomputed_linkage_rows=z,
    data_column_title="Protein",
    data_row_title="Protein",
)

b = PlotlyVisuBuilder(c, vertical_layout="h", horizontal_layout="h")
b.add_heatmap(_zmid="median")
# b.add_col_dendrogram()
# b.add_row_dendrogram()
# b.add_col_ticks(anchor_subplot="h", side="bottom")
# b.add_row_ticks(anchor_subplot="g", side="left")
b.get_figure().show()

# fig = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh").autobuild()
# fig.show()
