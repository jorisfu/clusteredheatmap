import numpy as np
import numpy.typing as npt
from typing import Literal

import scipy

from clusteredheatmap.types import DistFun, Vector, PDistFun

DistFunName = Literal["euclidean", "manhattan", "nan_euclidean"]
SCIPY_SUPPORTED_DISTANCES = [""]


def euclidean(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.linalg.norm(np.subtract(a, b)))


def euclidean_pds(a: Vector, b: Vector) -> np.float64:
    """
    Partial Distance Strategy as proposed by Dixon. See "Pattern Recognition with Partly Missing Data" by John K. Dixon.
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


def eirola_esd(a: Vector, b: Vector) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al. See [TODO]
    """

    # TODO: This returns a MATRIX!
    return np.array()

def manhattan(a: Vector, b: Vector) -> np.float64:
    return np.float64(np.subtract(a, b).sum())


_mapping: dict[DistFunName, DistFun] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "nan_euclidean": euclidean_pds,
}

_pdist_mapping: dict[DistFunName, PDistFun] = {
    "eirola_esd": eirola_esd,
}


def get_preferred_pdist_implementation(
    distance: DistFunName | DistFun,
) -> PDistFun:
    """
    From a given name of a distance function to use or a distance function, returns
    a callable function compatible with pdist to create 
    a condensed distance matrix from an array of observation
    vectors.
    """

    if callable(distance):
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)

    if distance in SCIPY_SUPPORTED_DISTANCES:
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)
    elif distance in _mapping.keys():
        dist_implementation = _mapping[distance]
        return lambda mat: scipy.spatial.distance.pdist(mat, dist_implementation)
    elif distance in _pdist_mapping.keys():
        return _pdist_mapping[distance]

    raise ValueError(f"Distance function {distance} not supported")

