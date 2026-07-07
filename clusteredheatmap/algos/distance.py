import numpy as np
from typing import Literal

import scipy

from clusteredheatmap.types import DistFun, Vector

DistFunName = Literal["euclidean", "manhattan", "nan_euclidean"]
SCIPY_SUPPORTED_DISTANCES = [""]


def euclidean(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.linalg.norm(np.subtract(a, b)))


def euclidean_pds(a: Vector, b: Vector) -> np.float64:
    """
    Partial Distance Strategy as proposed by Dixon. See 
    
    """
    nan_mask = np.isnan(a) | np.isnan(b)
    weight = len(nan_mask) / (len(nan_mask) - sum(nan_mask))

    masked_a = a[~nan_mask]
    masked_b = b[~nan_mask]

    d = scipy.spatial.distance.sqeuclidean(masked_a, masked_b)
    return np.float64(np.sqrt(weight * d))

def mesquita_expected_euclidean(a: Vector, b: Vector) -> np.float64:
    """
    Expected Euclidean Distance as proposed by Mesquita et al. See http://dx.doi.org/10.1016/j.neucom.2016.12.081

    Assumes distances are Nakagami-distributed. Data distribution modeled via a Gaussian mixture distribution.
    """

    # TODO
    return np.float64(0.0)

def manhattan(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.subtract(a, b).sum())


_mapping: dict[DistFunName, DistFun] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "nan_euclidean": euclidean_pds,
}


def get_preferred_implementation(
    distance: DistFunName | DistFun,
) -> DistFunName | DistFun:
    """
    From a given name of a distance function to use, returns
    either the name of the function if supported by scipy or the
    implementation of the method if not
    """

    if isinstance(distance, str):
        if distance in SCIPY_SUPPORTED_DISTANCES:
            return distance
        else:
            return _mapping[distance]

    elif callable(distance):
        return distance
