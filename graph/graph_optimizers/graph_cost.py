import matplotlib.pyplot as plt
import abc
import itertools
import math

import numpy as np
import scipy.stats
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

    def create_graph_metrics(self, matrix, **kwargs):
        return GraphMetrics(GraphDataset(None, self.distance_matrix.item), matrix, cost_boundaries=self.cost_boundaries)

    def _create_distance_matrix(self):
        mat = np.mat([[self.distance(i, j) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __calculate_total_cost(self, matrix):
        total_cost = 0
        graph_metrics = self.create_graph_metrics(matrix)
        w, r, f = 0, 0, 0
        if self.wiring_factor:
            wiring_cost = graph_metrics.wiring_cost()
            # total_cost += self.wiring_factor * wiring_cost.normalized_value
            w = wiring_cost.normalized_value
            self.cost_boundaries.wiring = wiring_cost.metric_boundaries
        if self.routing_factor:
            routing_cost = graph_metrics.routing_cost()
            # total_cost += self.routing_factor * routing_cost.normalized_value
            r = routing_cost.normalized_value
            self.cost_boundaries.routing = routing_cost.metric_boundaries
        if self.fuel_factor:
            fuel_cost = graph_metrics.fuel_cost()
            # total_cost += self.fuel_factor * fuel_cost.normalized_value
            f = fuel_cost.normalized_value
            self.cost_boundaries.fuel = fuel_cost.metric_boundaries
        total_cost = self.wiring_factor * (1 - w) ** 2 + self.routing_factor * (1 - r) ** 2 + self.fuel_factor * (1 - f) ** 2
        total_cost += 0.1 * (w ** 2 + r ** 2 + f ** 2)
        return - total_cost

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
    def distance(self, i, j):
        return perimeter_distance(i, j, self.num_nodes)


class GraphCostLattice(GraphCost):
    def distance(self, i, j):
        n = int(math.sqrt(self.num_nodes))
        i_location = i % n, int(i / n)
        j_location = j % n, int(j / n)
        return euclidean(i_location, j_location)


class GraphCostSphere(GraphCost):
    def _sphere_coordinates(self, theta, phi):
        x_0, y_0, z_0 = [0, 0, 0]
        r = 1
        x = x_0 + r * np.sin(theta) * np.cos(phi)
        y = y_0 + r * np.sin(theta) * np.sin(phi)
        z = z_0 + r * np.cos(theta)

        longitude = phi
        latitude = np.radians(90) - theta
        return x, y, z, longitude, latitude

    def _great_circle_dist(self, long1, lat1, long2, lat2, radius=1):
        delta_phi = lat1 - lat2
        delta_lambda = long2 - long1
        inner_sqrt = np.sin(delta_phi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2) ** 2
        result = 2 * radius * np.arcsin(np.sqrt(inner_sqrt))
        return result

    def sphere_distances(self, num_nodes):
        n = int(math.sqrt(num_nodes))
        theta_range = np.linspace(0, math.pi, n + 2)[1:-1]
        phi_range = np.linspace(0, 2 * math.pi, n + 1)
        points = list(itertools.product(theta_range, phi_range))
        points = np.mat([self._sphere_coordinates(*p) for p in points]).round(3)
        points = points[np.unique(points[:, range(3)], axis=0, return_index=True)[1], :]
        coords, lat_longs = points[:, range(3)], points[:, [3, 4]]
        dists = np.mat([[self._great_circle_dist(*i, *j) for j in lat_longs] for i in lat_longs.tolist()])
        return coords, dists

    def _create_distance_matrix(self):
        self.coords, self.dist = self.sphere_distances(self.num_nodes)
        return self.dist

    def distance(self, i, j):
        return self.dist[i, j]


class GraphCostTorus(GraphCost):
    def distance(self, i, j):
        n = int(math.sqrt(self.num_nodes))
        i_location = i % n, int(i / n)
        j_location = j % n, int(j / n)
        x = min([abs(i_location[0] - j_location[0]), n - abs(i_location[0] - j_location[0])])
        y = min([abs(i_location[1] - j_location[1]), n - abs(i_location[1] - j_location[1])])
        return np.sqrt(x ** 2 + y ** 2)


class GraphCostFacade:
    type_mapping = {'circular': GraphCostCircular, 'lattice': GraphCostLattice,
                    'sphere': GraphCostSphere, 'torus': GraphCostTorus}

    def get_cost(self, num_nodes, wiring_factor, routing_factor, fuel_factor, method, type, *args, **kwargs):
        cost_class = self.type_mapping[type]
        return cost_class(num_nodes, wiring_factor, routing_factor, fuel_factor, method, *args, **kwargs)


if __name__ == '__main__':
    tr = GraphCostTorus(100, 1, 1, 1, 'maximize')
