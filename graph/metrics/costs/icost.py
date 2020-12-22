import random
from abc import ABC
from typing import Union

import igraph
import numpy as np
import pandas as pd
import scipy.sparse.csgraph

from graph.metrics.Metric import MetricBoundaries


class ICost(ABC):
    def __init__(self, number_of_nodes: int, number_of_edges: int,
                 adjacency: np.mat, distances: np.mat, positions: callable, graph: igraph.Graph, boundaries: Union[MetricBoundaries, None]):
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        self.adjacency = adjacency
        self.distances = distances
        self.positions = positions
        self.graph = graph
        self._boundaries = boundaries or MetricBoundaries()

    def __group_by_matrix(self, data: np.ndarray):
        unique_elements, counts = np.unique(data, return_counts=True)
        data = np.mat([unique_elements, counts]).transpose()
        grouped_by = pd.DataFrame(data, columns=['dist', 'count'])
        grouped_by = grouped_by[grouped_by['dist'] > 0]
        return grouped_by

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        indices = random.sample(range(self.number_of_nodes), 1000) if self.number_of_nodes > 1000 else None
        if self.graph is not None:
            all_pairs_shortest_path = self.graph.shortest_paths(indices, weights='weight' if weight else None)
        else:
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.adjacency, directed=False,
                                                                         unweighted=not weight, indices=indices)
        gb = self.__group_by_matrix(all_pairs_shortest_path)
        return gb

    def cost(self):
        raise NotImplementedError()

    @property
    def boundaries(self) -> MetricBoundaries:
        raise NotImplementedError()
