from typing import Literal

from clusteredheatmap.algos.distance import DistFunName
from clusteredheatmap.types import ClusteringFun, DistFun, LinkageFun
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


def get_preferred_implementation(linkage: LinkageFunName | LinkageFun) -> LinkageFun:
    if callable(linkage):
        return linkage

    if linkage in SCIPY_SUPPORTED_LINKAGES:
        fun: LinkageFun = lambda distmat: scipy.cluster.hierarchy.linkage(
            distmat,
            method=linkage,
            optimal_ordering=False,
        )

        return fun

    else:
        raise ValueError(f"Linkage method '{linkage}' not supported")
