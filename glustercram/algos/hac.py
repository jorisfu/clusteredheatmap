from collections.abc import Iterable
from glustercram.types import T, DistFun, LinkageFun

inf = float("inf")


def general(data: Iterable[T], distance: DistFun[T], linkage: LinkageFun[T]):
    """
    Trivial O(n³) algorithm for HAC as described in Section 8.2.1
    TODO: Full citation
    """
    return
