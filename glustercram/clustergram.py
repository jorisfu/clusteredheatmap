# pyright: reportExplicitAny=false

from typing import Any
from glustercram.algos import hac
from glustercram.algos.distance import DistFunName
from glustercram.algos.linkage import LinkageFunName
from glustercram.dendrogram import Dendrogram
from glustercram.types import T, ClusteringFun, DistFun, LinkageFun
import glustercram.algos.distance as dist
import glustercram.algos.linkage as link
import pandas as pd
import scipy

class Clustergram:
    def __init__(
        self,
        data: pd.DataFrame,
        distance: DistFunName | DistFun,
        linkage: LinkageFunName | LinkageFun,
    ) -> None:
        self.data: pd.DataFrame = data
        self.data_rows = self.data.to_numpy()
        self.data_cols = self.data.T.to_numpy()

        """ Linkage + Distance method that performs the clustering """
        self.calc_method: ClusteringFun = link.get_preferred_implementation(linkage, distance)

        self.linkage_matrix_rows = self.calc_method(self.data_rows)
        self.linkage_matrix_cols = self.calc_method(self.data_cols)


    def get_visualization_plotly(self):

        return None
