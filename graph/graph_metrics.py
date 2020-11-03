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
        self.mean_edge_distance = None

    def __group_by_matrix(self, mat: np.ndarray):
        gb = pd.DataFrame(mat.flatten())
        gb.columns = ['dist']
        gb['count'] = 0
        gb = gb[gb['dist'] < float('inf')]
        gb = gb.groupby(['dist'], as_index=False).agg({'count': 'count'})
        return gb

    def __mean_edge_distance(self):
        if self.mean_edge_distance is None:
            random_starts = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            random_ends = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            mean_distance = np.array([self.distances(u, v) for u in random_starts for v in random_ends if u != v]).mean()
            self.mean_edge_distance = mean_distance
        return self.mean_edge_distance

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
        logger.debug('start wiring cost')
        mean_weight = self.__mean_edge_distance()
        expected_in_random_network = mean_weight * self.number_of_edges
        result = MetricResult(self.sparse_matrix.sum() / 2, expected_in_random_network)
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        expected_in_random_network = math.log(self.number_of_nodes) / math.log(mean_degree)
        df = self.all_path_lengths(False)
        result = MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(), expected_in_random_network)
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        mean_weight = self.__mean_edge_distance()
        expected_routing_cost = math.log(self.number_of_nodes) / math.log(mean_degree)
        expected_in_random_netowrk = mean_weight * expected_routing_cost
        df = self.all_path_lengths(True)
        result = MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(), expected_in_random_netowrk)
        logger.debug('end fuel cost')
        return result
