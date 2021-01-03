import random
from abc import ABC
from typing import Union

import igraph
import numpy as np
import pandas as pd
import scipy.sparse.csgraph

from graph.graph_dataset import GraphDataset
from graph.metrics.Metric import MetricBoundaries


class ICost(ABC):
    def __init__(self, graph_dataset: GraphDataset, boundaries: Union[MetricBoundaries] = None):
        self.graph_dataset = graph_dataset
        self._boundaries = boundaries or MetricBoundaries()

    def __group_by_matrix(self, data: np.ndarray):
        unique_elements, counts = np.unique(data, return_counts=True)
        data = np.mat([unique_elements, counts]).transpose()
        grouped_by = pd.DataFrame(data, columns=['dist', 'count'])
        grouped_by = grouped_by[grouped_by['dist'] > 0]
        return grouped_by

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        indices = random.sample(range(self.graph_dataset.number_of_nodes), 1000) if self.graph_dataset.number_of_nodes > 1000 else None
        if self.graph_dataset.graph is not None:
            all_pairs_shortest_path = self.graph_dataset.graph.shortest_paths(indices, weights='weight' if weight else None)
        else:
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.graph_dataset.adjacency, directed=False,
                                                                         unweighted=not weight, indices=indices)
        gb = self.__group_by_matrix(all_pairs_shortest_path)
        return gb

    def cost(self):
        raise NotImplementedError()

    @property
    def boundaries(self) -> MetricBoundaries:
        raise NotImplementedError()
