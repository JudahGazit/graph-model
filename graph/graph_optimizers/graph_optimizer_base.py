import abc
import random

import networkx as nx
import numpy as np

from graph.graph_dataset import GraphDataset
from graph.metrics.graph_cost import GraphCostFacade


class GraphOptimizerBase(abc.ABC):
    def __init__(self, num_nodes, num_edges, factors, method, cost_type):
        self.graph_cost = GraphCostFacade().get_cost(num_nodes, factors, method, cost_type)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self._total_possible_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)

    def _optimal_matrix(self):
        raise NotImplementedError()

    def randomize_edges(self):
        edges_indices = set(random.sample(range(self._total_possible_edges), self.num_edges))
        edges_vec = [1 if i in edges_indices else 0 for i in range(self._total_possible_edges)]
        return edges_vec

    def optimize(self):
        min_arg = self._optimal_matrix()
        result = np.multiply(self.graph_cost.distance_matrix, min_arg)
        graph = nx.from_numpy_matrix(result)
        return GraphDataset(graph, self.graph_cost.distance_matrix, self.graph_cost.position)