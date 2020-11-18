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
    def __init__(self, graph_dataset=None, matrix=None, topology='circular',
                 optimal_wiring_cost=None, worst_wiring_cost=None,
                 optimal_fuel_cost=None, worst_fuel_cost=None):
        self.topology = topology
        self.graph = graph_dataset.graph if graph_dataset else None  # nx.Graph
        self.distances = graph_dataset.distances if graph_dataset else None
        self.all_shortest_paths = {True: None, False: None}
        self._optimal_wiring_cost = optimal_wiring_cost
        self._worst_wiring_cost = worst_wiring_cost
        self._optimal_fuel_cost = optimal_fuel_cost
        self._worst_fuel_cost = worst_fuel_cost

        if self.graph:
            self.number_of_nodes = self.graph.number_of_nodes()
            self.number_of_edges = self.graph.number_of_edges()
        else:
            self.number_of_nodes = matrix.shape[0]
            self.number_of_edges = int(np.count_nonzero(matrix) / 2)

        if matrix is not None:
            self.sparse_matrix = matrix
            self.igraph = igraph.Graph.Weighted_Adjacency(matrix.tolist(), mode=igraph.ADJ_UNDIRECTED) if self.number_of_nodes <= 500 else None
        else:
            self.sparse_matrix = nx.to_scipy_sparse_matrix(graph_dataset.graph)
            self.igraph = igraph.Graph.from_networkx(self.graph) if self.number_of_nodes <= 500 else None


    def __group_by_matrix(self, mat: np.ndarray):
        unique_elements, counts = np.unique(mat, return_counts=True)
        mat = np.mat([unique_elements, counts]).transpose()
        gb = pd.DataFrame(mat, columns=['dist', 'count'])
        gb = gb[gb['dist'] > 0]
        return gb

    def __mean_distance(self, distance_method):
        random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
        mean_distance = np.array([distance_method(u, v) for u, v in itertools.combinations(random_nodes, 2)]).mean()
        return mean_distance

    def __top_edges_by_length(self, reverse=False):
        total_number_of_possible_edges = self.number_of_nodes * (self.number_of_nodes - 1) / 2
        percentile = self.number_of_edges / total_number_of_possible_edges
        random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
        distances = sorted([self.distances(u, v) for u, v in itertools.combinations(random_nodes, 2)], reverse=reverse)
        number_of_random_edges = len(distances)
        sum_of_percentile_samples = sum(distances[:(max([int(number_of_random_edges * percentile), 1]))])
        return sum_of_percentile_samples * (total_number_of_possible_edges / number_of_random_edges)

    @property
    def optimal_fuel_cost(self):
        if self._optimal_fuel_cost is None:
            self._optimal_fuel_cost = self.__mean_distance(self.distances)
        return self._optimal_fuel_cost

    @property
    def worst_fuel_cost(self):
        if self._worst_fuel_cost is None:
            n = int(math.sqrt(self.number_of_nodes))
            self._worst_fuel_cost = self.__mean_distance(lambda u, v: manhatten(u, v, n))
        return self._worst_fuel_cost

    @property
    def optimal_wiring_cost(self):
        if self._optimal_wiring_cost is None:
            self._optimal_wiring_cost = self.__top_edges_by_length(reverse=False)
        return self._optimal_wiring_cost

    @property
    def worst_wiring_cost(self):
        if self._worst_wiring_cost is None:
            self._worst_wiring_cost = self.__top_edges_by_length(reverse=True)
        return self._worst_wiring_cost

    @property
    def optimal_routing_cost(self):
        number_of_pairs = self.number_of_nodes * (self.number_of_nodes - 1) / 2
        number_of_pairs_in_distance_1 = self.number_of_edges
        number_of_pairs_in_distance_2 = number_of_pairs - self.number_of_edges
        mean_degree = (number_of_pairs_in_distance_1 + 2 * number_of_pairs_in_distance_2) / number_of_pairs
        return mean_degree

    @property
    def mean_routing_cost(self):
        mean_degree = 2 * self.number_of_edges / self.number_of_nodes
        expected_in_random_network = math.log(self.number_of_nodes) / math.log(mean_degree)
        return expected_in_random_network

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        if self.all_shortest_paths[weight] is None:
            logger.debug(f'shortest path started - weight={weight}')
            indices = random.sample(range(self.number_of_nodes), 1000) if self.number_of_nodes > 1000 else None
            if self.igraph is not None:
                all_pairs_shortest_path = self.igraph.shortest_paths(indices, weights='weight' if weight else None)
            else:
                all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                             unweighted=not weight, indices=indices)
            logger.debug('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            logger.debug('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def wiring_cost(self):
        logger.debug('start wiring cost')
        result = MetricResult(self.sparse_matrix.sum() / 2, self.optimal_wiring_cost, self.worst_wiring_cost)
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        df = self.all_path_lengths(False)
        routing_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = MetricResult(routing_cost, self.optimal_routing_cost, mean_value=self.mean_routing_cost)
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        df = self.all_path_lengths(True)
        fuel_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = MetricResult(fuel_cost, self.optimal_fuel_cost, self.worst_fuel_cost)
        logger.debug('end fuel cost')
        return result
