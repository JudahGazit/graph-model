import abc
import math

import numpy as np
from scipy.spatial.distance import euclidean

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset
from graph.graph_metrics import GraphMetrics, CostBoundaries


class GraphCost(abc.ABC):
    def __init__(self, num_nodes, wiring_factor, routing_factor, fuel_factor, method):
        self.num_nodes = num_nodes
        self.wiring_factor = wiring_factor
        self.fuel_factor = fuel_factor
        self.method = method
        self.distance_matrix = self._create_distance_matrix()
        self.routing_factor = routing_factor
        self.cost_boundaries = CostBoundaries()

    def distance(self, i, j):
        raise NotImplementedError()

    def create_graph_metrics(self, matrix, **kwargs) -> GraphMetrics:
        raise NotImplementedError()

    def _create_distance_matrix(self):
        mat = np.mat([[self.distance(i, j) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __calculate_total_cost(self, matrix):
        total_cost = 0
        graph_metrics = self.create_graph_metrics(matrix)
        if self.wiring_factor:
            wiring_cost = graph_metrics.wiring_cost()
            total_cost += self.wiring_factor * wiring_cost.normalized_value
            self.cost_boundaries.wiring = wiring_cost.metric_boundaries
        if self.routing_factor:
            routing_cost = graph_metrics.routing_cost()
            total_cost += self.routing_factor * routing_cost.normalized_value
            self.cost_boundaries.routing = routing_cost.metric_boundaries
        if self.fuel_factor:
            fuel_cost = graph_metrics.fuel_cost()
            total_cost += self.fuel_factor * fuel_cost.normalized_value
            self.cost_boundaries.fuel = fuel_cost.metric_boundaries
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
        cost = float('inf') if math.isinf(total_cost) else method_factor * total_cost
        return cost

    def triangular_to_mat(self, triangular_as_vec):
        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int)
        mat[np.triu_indices(self.num_nodes, 1)] = triangular_as_vec
        return mat + mat.transpose()


class GraphCostCircular(GraphCost):
    def create_graph_metrics(self, matrix, **kwargs):
        return GraphMetrics(GraphDataset(None, self.distance_matrix.item), matrix, topology='circular',
                            cost_boundaries=self.cost_boundaries, **kwargs)

    def distance(self, i, j):
        return perimeter_distance(i, j, self.num_nodes)



class GraphCostLattice(GraphCost):
    def create_graph_metrics(self, matrix, **kwargs):
        return GraphMetrics(GraphDataset(None, self.distance_matrix.item), matrix, topology='lattice',
                            cost_boundaries=self.cost_boundaries,
                            **kwargs)

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
