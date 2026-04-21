from glustercram.types import DistFun, T


def single(distance: DistFun[T], a: set[T], b: set[T]) -> float:
    dists: list[float] = []
    for x in a:
        for y in b:
            dists.append(distance(x, y))

    return min(dists)
