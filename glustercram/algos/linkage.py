from glustercram.types import DistFun, T


from typing import Literal

LinkageFunName = Literal["single"]


def single(distance: DistFun[T], a: set[T], b: set[T]) -> float:
    dists: list[float] = []
    for x in a:
        for y in b:
            dists.append(distance(x, y))

    return min(dists)
