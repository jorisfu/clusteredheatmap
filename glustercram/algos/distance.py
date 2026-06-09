import numpy as np
from typing import Literal

from glustercram.types import DistFun, Vector

DistFunName = Literal["euclidean", "manhattan"]
SCIPY_SUPPORTED_DISTANCES = [""]


def euclidean(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.linalg.norm(np.subtract(a, b)))


def manhattan(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.subtract(a, b).sum())


_mapping: dict[DistFunName, DistFun] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
}


def get_distfun_for_scipy(distance: DistFunName):
    """
    From a given name of a distance function to use, returns
    either the name of the function if supported by scipy or the
    implementation of the method if not
    """

    if distance in SCIPY_SUPPORTED_DISTANCES:
        return distance
    else:
        return _mapping[distance]
