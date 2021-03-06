import abc
import math

import numpy as np
from scipy.spatial.distance import euclidean

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset
from graph.graph_metrics import GraphMetrics


class GraphCost(abc.ABC):
    def __init__(self, num_nodes, wiring_factor, routing_factor, fuel_factor, method):
        self.num_nodes = num_nodes
        self.wiring_factor = wiring_factor
        self.fuel_factor = fuel_factor
        self.method = method
        self.distance_matrix = self._create_distance_matrix()
        self.routing_factor = routing_factor
        self.optimal_fuel_cost = None
        self.optimal_wiring_cost = None

    def distance(self, i, j):
        raise NotImplementedError()

    def _create_graph_metrics(self, matrix, **kwargs) -> GraphMetrics:
        raise NotImplementedError()

    def _create_distance_matrix(self):
        mat = np.mat([[self.distance(i, j) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __calculate_total_cost(self, matrix):
        total_cost = 0
        graph_metrics = self._create_graph_metrics(matrix,
                                                   optimal_fuel_cost=self.optimal_fuel_cost,
                                                   optimal_wiring_cost=self.optimal_wiring_cost)
        if self.wiring_factor:
            wiring_cost = graph_metrics.wiring_cost()
            total_cost += self.wiring_factor * wiring_cost.normalized_value
            self.optimal_wiring_cost = wiring_cost.normalization_factor
        if self.routing_factor:
            total_cost += self.routing_factor * graph_metrics.routing_cost().normalized_value
        if self.fuel_factor:
            fuel_cost = graph_metrics.fuel_cost()
            total_cost += self.fuel_factor * fuel_cost.normalized_value
            self.optimal_fuel_cost = fuel_cost.normalization_factor
        return total_cost

    def triangular_index(self, i, row_index=0):
        num_rows = self.num_nodes - (row_index + 1)
        if i < num_rows:
            return 0, i + 1
        res = self.triangular_index(i - num_rows, row_index + 1)
        return res[0] + 1, res[1] + 1

    def cost(self, mat):
        matrix = np.multiply(self.distance_matrix, mat)
        total_cost = self.__calculate_total_cost(matrix)
        method_factor = 1 if self.method == 'minimize' else -1
        return method_factor * total_cost

    def triangular_to_mat(self, triangular_as_vec):
        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int)
        mat[np.triu_indices(self.num_nodes, 1)] = triangular_as_vec
        return mat + mat.transpose()


class GraphCostCircular(GraphCost):
    def _create_graph_metrics(self, matrix, **kwargs):
        return GraphMetrics(GraphDataset(None, self.distance_matrix.item), matrix, topology='circular', **kwargs)

    def distance(self, i, j):
        return perimeter_distance(i, j, self.num_nodes)



class GraphCostLattice(GraphCost):
    def _create_graph_metrics(self, matrix, **kwargs):
        return GraphMetrics(GraphDataset(None, self.distance_matrix.item), matrix, topology='lattice', **kwargs)

    def distance(self, i, j):
        n = int(math.sqrt(self.num_nodes))
        i_location = i % n, int(i / n)
        j_location = j % n, int(j / n)
        return euclidean(i_location, j_location)


class GraphCostFacade:
    type_mapping = {'circular': GraphCostCircular, 'lattice': GraphCostLattice}

    def get_cost(self, num_nodes, wiring_factor, routing_factor, fuel_factor, method, type, *args, **kwargs):
        cost_class = self.type_mapping[type]
        return cost_class(num_nodes, wiring_factor, routing_factor, fuel_factor, method, *args, **kwargs)
