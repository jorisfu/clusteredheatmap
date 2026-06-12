import numpy as np
from typing import Literal

import scipy

from glustercram.types import DistFun, Vector

DistFunName = Literal["euclidean", "manhattan", "nan_euclidean"]
SCIPY_SUPPORTED_DISTANCES = [""]


def euclidean(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.linalg.norm(np.subtract(a, b)))


def nan_euclidean(a: Vector, b: Vector) -> np.float64:
    """
    Literally the simplest approach, just mask the NaN parts and compute the distance from the rest.
    Same thing that https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html does

    TODO: Look at their citation
    """
    nan_mask = np.isnan(a) | np.isnan(b)
    weight = len(nan_mask) / (len(nan_mask) - sum(nan_mask))

    masked_a = a[~nan_mask]
    masked_b = b[~nan_mask]

    d = scipy.spatial.distance.sqeuclidean(masked_a, masked_b)
    return np.float64(np.sqrt(weight * d))


def manhattan(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.subtract(a, b).sum())


_mapping: dict[DistFunName, DistFun] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "nan_euclidean": nan_euclidean,
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
