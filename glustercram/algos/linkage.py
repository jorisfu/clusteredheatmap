from typing import Literal

from glustercram.algos.distance import DistFunName, get_distfun_for_scipy
from glustercram.types import ClusteringFun, DistFun, LinkageFun
import scipy

LinkageFunName = Literal[
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
]


SCIPY_SUPPORTED_LINKAGES = [
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
]


def get_preferred_implementation(
    linkage: LinkageFunName | LinkageFun, distance: DistFunName | DistFun
) -> ClusteringFun:
    # TODO
    if not isinstance(linkage, str):
        raise ValueError("Custom linkage method not supported (yet)")

    if linkage in SCIPY_SUPPORTED_LINKAGES:
        distance_for_scipy = (
            distance if callable(distance) else get_distfun_for_scipy(distance)
        )

        fun: ClusteringFun = lambda data: scipy.cluster.hierarchy.linkage(
            data,
            method=linkage,
            metric=distance_for_scipy,
            optimal_ordering=False,
        )

        return fun

    else:
        raise ValueError(f"Linkage method '{linkage}' not supported")
