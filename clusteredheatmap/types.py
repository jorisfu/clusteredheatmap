from typing import Callable, NamedTuple
import numpy as np
import numpy.typing as npt

""" List of coordinates to be mapped in euclidean space """
Vector = npt.ArrayLike

""" Distance function measuring distance between two data points """
DistFun = Callable[..., np.float64]  # Ellipsis to allow for kwargs

""" Distance function operating on a matrix and returning a matrix, compatible with scipy pdist """
PDistFun = Callable[..., npt.NDArray[np.float64]]  # Ellipsis to allow for kwargs

""" Function generating a linkage matrix from a distance matrix, compatible with scipy.cluster.hierarchy.linkage """
LinkageFun = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

""" Function computing a linkage matrix from a given array of observations with a predefined linkage and distance method """
ClusteringFun = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

""" 2D matrix with floats (np.nan allowed) """
HeatmapMatrix = npt.NDArray[np.float64]

""" Plotly-compatible color definition"""
Color = str

""" Plotly-compatible colorscale definition """
Colorscale = list[list[float | Color]]


class LayoutPoint(NamedTuple):
    """Point in 2D space, used for layouting"""

    row: int
    col: int
