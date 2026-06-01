#pyright: reportExplicitAny=false

from typing import Any
from glustercram.algos import hac
from glustercram.algos.distance import DistFunName
from glustercram.algos.linkage import LinkageFunName
from glustercram.dendrogram import Dendrogram
from glustercram.types import T, DistFun, LinkageFun
import glustercram.algos.distance as dist
import glustercram.algos.linkage as link
import pandas as pd

def get_distfun(
    distance: DistFunName | DistFun[Any],
) -> DistFun[Any]:
    if isinstance(distance, str):
        match distance:
            case "euclidean":
                return dist.euclidean
            case "manhattan":
                return dist.manhattan

    else:
        return distance

def get_linkagefun(
    linkage: LinkageFunName | LinkageFun[Any],
) -> LinkageFun[Any]:
    if isinstance(linkage, str):
        match linkage:
            case "single":
                return link.single
    else:
        return linkage

class Clustergram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance: DistFunName | DistFun[T],
        linkage: LinkageFunName | LinkageFun[T],
    ) -> None:
        self.data: pd.DataFrame = data
        
        self.distance: DistFun[Any] = get_distfun(distance)
        self.linkage: LinkageFun[Any] = get_linkagefun(linkage)

        # Taking each row of the df as a vector
        self.dendro_rows: Dendrogram = None # TODO

        # Taking each column of the df as a vector
        self.dendro_cols: Dendrogram = None # TODO


    def get_visualization_plotly(self):
        
        
        return None
