from typing import Callable, TypeVar
import numpy.typing as npt

""" Generic type used for the data to cluster """
T = TypeVar("T")

""" Distance function measuring distance between two data points """
DistFun = Callable[[T, T], float]

""" 
Linkage function measuring distance between two sets of data points. 
Requires a corresponding distance function.
"""
LinkageFun = Callable[[DistFun[T], set[T], set[T]], float]

""" List of coordinates to be mapped in euclidean space """
Vector = npt.ArrayLike
