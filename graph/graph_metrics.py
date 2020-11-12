import itertools
import logging
import math
import random

import igraph
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock
import scipy.sparse.csgraph

from graph.metric_result import MetricResult

logger = logging.getLogger('metrics')


def manhatten(u, v, n):
    u = u % n, int(u / n)
    v = v % n, int(v / n)
    return cityblock(u, v)


class GraphMetrics:
    def __init__(self, graph_dataset=None, matrix=None, topology='circular', optimal_wiring_cost=None,
                 optimal_fuel_cost=None):
        self.topology = topology
        self.graph = graph_dataset.graph if graph_dataset else None  # nx.Graph
        self.distances = graph_dataset.distances if graph_dataset else None
        self.all_shortest_paths = {True: None, False: None}
        self._optimal_wiring_cost = optimal_wiring_cost
        self._optimal_fuel_cost = optimal_fuel_cost

        if matrix is not None:
            self.sparse_matrix = matrix
        else:
            self.sparse_matrix = nx.to_scipy_sparse_matrix(graph_dataset.graph)

        if self.graph:
            self.number_of_nodes = self.graph.number_of_nodes()
            self.number_of_edges = self.graph.number_of_edges()
        else:
            self.number_of_nodes = matrix.shape[0]
            self.number_of_edges = int(np.count_nonzero(matrix) / 2)


    def __group_by_matrix(self, mat: np.ndarray):
        unique_elements, counts = np.unique(mat, return_counts=True)
        mat = np.mat([unique_elements, counts]).transpose()
        gb = pd.DataFrame(mat, columns=['dist', 'count'])
        gb = gb[(gb['dist'] > 0) & (gb['dist'] < float('inf'))]
        return gb

    @property
    def optimal_fuel_cost(self):
        if self._optimal_fuel_cost is None:
            random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            mean_distance = np.array([self.distances(u, v) for u, v in itertools.combinations(random_nodes, 2)]).mean()
            self._optimal_fuel_cost = mean_distance
        return self._optimal_fuel_cost

    @property
    def optimal_wiring_cost(self):
        if self._optimal_wiring_cost is None:
            total_number_of_possible_edges = self.number_of_nodes * (self.number_of_nodes - 1) / 2
            percentile = self.number_of_edges / total_number_of_possible_edges
            random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            distances = sorted([self.distances(u, v) for u, v in itertools.combinations(random_nodes, 2)])
            number_of_random_edges = len(distances)
            sum_of_percentile_samples = sum(distances[:(max([int(number_of_random_edges * percentile), 1]))])
            self._optimal_wiring_cost = sum_of_percentile_samples * (total_number_of_possible_edges / number_of_random_edges)
        return self._optimal_wiring_cost

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        if self.all_shortest_paths[weight] is None:
            logger.debug(f'shortest path started - weight={weight}')
            indices = random.sample(range(self.number_of_nodes), 1000) if self.number_of_nodes > 1000 else None
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                         unweighted=not weight, indices=indices)
            logger.debug('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            logger.debug('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def wiring_cost(self):
        logger.debug('start wiring cost')
        result = MetricResult(self.sparse_matrix.sum() / 2, self.optimal_wiring_cost)
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        expected_in_random_network = math.log(self.number_of_nodes) / math.log(mean_degree)
        df = self.all_path_lengths(False)
        routing_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = MetricResult(routing_cost, expected_in_random_network, 1 - 0.5 * routing_cost / expected_in_random_network)
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        df = self.all_path_lengths(True)
        result = MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(), self.optimal_fuel_cost)
        logger.debug('end fuel cost')
        return result
