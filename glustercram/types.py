from collections import namedtuple
from typing import Any, Callable, NamedTuple
import numpy as np
import numpy.typing as npt

""" List of coordinates to be mapped in euclidean space """
Vector = npt.ArrayLike

""" Distance function measuring distance between two data points """
DistFun = Callable[[Vector, Vector], np.float64]

""" Function generating a linkage matrix, compatible with scipy.cluster.hierarchy.linkage """
LinkageFun = Callable[[npt.NDArray[np.float64], DistFun], npt.NDArray[np.float64]]

""" Function computing a linkage matrix from a given array of observations with a predefined linkage and distance method """
ClusteringFun = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

""" 2D matrix with floats (np.nan allowed) """
HeatmapMatrix = npt.NDArray[np.float64]

""" Plotly-compatible color definition"""
Color = str

""" Point in 2D space, used for layouting """


class LayoutPoint(NamedTuple):
    row: int
    col: int
