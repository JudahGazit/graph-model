import logging

import igraph
import networkx as nx
import numpy as np

from graph.metrics.Metric import MetricBoundaries
from graph.metrics.costs.fuel_cost import FuelCost
from graph.metrics.costs.intersection_cost import IntersectionCost
from graph.metrics.costs.routing_cost import RoutingCost
from graph.metrics.costs.wiring_cost import WiringCost

logger = logging.getLogger('metrics')


class CostBoundaries:
    def __init__(self, wiring=None, routing=None, fuel=None):
        self.wiring = wiring or MetricBoundaries()
        self.routing = routing or MetricBoundaries()
        self.fuel = fuel or MetricBoundaries()


class GraphMetrics:
    def __init__(self, graph_dataset=None, matrix=None, cost_boundaries=None):
        self.__initiate_data(graph_dataset, matrix)
        self.cost_boundaries = cost_boundaries or CostBoundaries()
        self._wiring = WiringCost(self.number_of_nodes, self.number_of_edges, self.adjacency, self.distances, self.positions, self.graph, self.cost_boundaries.wiring)
        self._routing = RoutingCost(self.number_of_nodes, self.number_of_edges, self.adjacency, self.distances, self.positions, self.graph, self.cost_boundaries.routing)
        self._fuel = FuelCost(self.number_of_nodes, self.number_of_edges, self.adjacency, self.distances, self.positions, self.graph, self.cost_boundaries.fuel)
        self._intersection = IntersectionCost(self.number_of_nodes, self.number_of_edges, self.adjacency, self.distances, self.positions, self.graph, None)
        self.cost_boundaries = self.__update_cost_boundaries(cost_boundaries)

    def __initiate_data(self, graph_dataset, matrix):
        graph = graph_dataset.graph if graph_dataset else None  # nx.Graph
        self.positions = graph_dataset.positions if graph_dataset else None
        if graph:
            self.number_of_nodes = graph.number_of_nodes()
            self.number_of_edges = graph.number_of_edges()
        else:
            self.number_of_nodes = matrix.shape[0]
            self.number_of_edges = int(np.count_nonzero(matrix) / 2)
        self.distances = np.mat([[graph_dataset.distances(i, j) if i != j else 0
                                  for j in range(self.number_of_nodes)]
                                 for i in range(self.number_of_nodes)])
        if matrix is not None:
            self.adjacency = matrix
            self.graph = igraph.Graph.Weighted_Adjacency(matrix.tolist(), mode=igraph.ADJ_UNDIRECTED) \
                if self.number_of_nodes <= 500 else None
        else:
            self.adjacency = nx.to_numpy_matrix(graph)
            self.graph = igraph.Graph.from_networkx(graph) if self.number_of_nodes <= 500 else None

    def __update_cost_boundaries(self, cost_boundaries):
        cost_boundaries = cost_boundaries if cost_boundaries is not None else CostBoundaries()
        cost_boundaries.wiring = self._wiring.boundaries
        cost_boundaries.routing = self._routing.boundaries
        cost_boundaries.fuel = self._fuel.boundaries
        return cost_boundaries

    def all_path_lengths(self, weight: bool):
        return self._routing.all_path_lengths(weight)

    def wiring_cost(self):
        logger.debug('start wiring cost')
        result = self._wiring.cost()
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        result = self._routing.cost()
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        result = self._fuel.cost()
        logger.debug('end fuel cost')
        return result

    def collision_cost(self, grid_scale=2):
        logger.debug('start intersection cost')
        result = self._intersection.cost()
        logger.debug('end intersection cost')
        return result