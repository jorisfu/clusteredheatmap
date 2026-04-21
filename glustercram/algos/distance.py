import numpy as np

from glustercram.types import Vector


def euclidean(a: Vector, b: Vector) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def manhattan(a: Vector, b: Vector) -> float:
    return float(np.subtract(a, b).sum())
