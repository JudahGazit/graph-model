import logging
import math
import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse

from graph.metric_result import MetricResult

logger = logging.getLogger('main')


class GraphMetrics:
    def __init__(self, graph_dataset=None, matrix=None):
        self.graph = graph_dataset.graph if graph_dataset else None
        self.distances = graph_dataset.distances if graph_dataset else None
        self.sparse_matrix = matrix if matrix is not None else nx.to_scipy_sparse_matrix(graph_dataset.graph)
        self.number_of_nodes = self.graph.number_of_nodes() if self.graph else self.sparse_matrix.shape[0]
        self.number_of_edges = self.graph.number_of_edges() if self.graph else np.count_nonzero(self.sparse_matrix)
        self.all_shortest_paths = {True: None, False: None}

    def __group_by_matrix(self, mat: np.ndarray):
        gb = pd.DataFrame(mat.flatten())
        gb.columns = ['dist']
        gb['count'] = 0
        gb = gb[gb['dist'] < float('inf')]
        gb = gb.groupby(['dist'], as_index=False).agg({'count': 'count'})
        return gb

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        if self.all_shortest_paths[weight] is None:
            logger.debug(f'shortest path started - weight={weight}')
            indices = random.sample(range(self.number_of_nodes), min([1000, self.number_of_nodes]))
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                         unweighted=not weight, indices=indices)
            logger.debug('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            logger.debug('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def wiring_cost(self):
        mean_weight = np.array(
            [self.distances(u, v) for u in range(self.number_of_nodes) for v in range(self.number_of_nodes) if
             u != v]).mean()
        expected_in_random_network = mean_weight * self.number_of_edges
        return MetricResult(self.sparse_matrix.sum() / 2, expected_in_random_network)

    def routing_cost(self):
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        expected_in_random_network = math.log(self.number_of_nodes) / math.log(mean_degree)
        df = self.all_path_lengths(False)
        return MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(), expected_in_random_network)

    def fuel_cost(self):
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        mean_weight = np.array(
            [self.distances(u, v) for u in range(self.number_of_nodes) for v in range(self.number_of_nodes) if
             u != v]).mean()
        expected_routing_cost = math.log(self.number_of_nodes) / math.log(mean_degree)
        expected_in_random_netowrk = mean_weight * expected_routing_cost
        df = self.all_path_lengths(True)
        return MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(),
                            expected_in_random_netowrk)
