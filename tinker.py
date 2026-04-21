# Random tests
import numpy as np

from glustercram.algos import hac
from glustercram.types import Vector

from glustercram.algos import distance, linkage

points: list[Vector] = [
    (1, 1),
    (1.5, 1),
    (1, 1.5),
    (3.5, 1),
    (3.5, 2),
]

print(hac.general(points, distance.euclidean, linkage.single))
