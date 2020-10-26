import abc

import networkx as nx
import numpy as np

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset
from graph.graph_optimizers.graph_cost import GraphCost


class GraphOptimizerBase(abc.ABC):
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method='minimize'):
        self.graph_cost = GraphCost(num_nodes, wiring_factor, routing_factor, fuel_factor, method)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self._total_possible_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)

    def _optimal_matrix(self):
        raise NotImplementedError()

    def optimize(self):
        min_arg = self._optimal_matrix()
        result = np.multiply(self.graph_cost.distance_matrix, min_arg)
        graph = nx.from_numpy_matrix(result)
        return GraphDataset(graph, lambda u, v: perimeter_distance(u, v, self.num_nodes))