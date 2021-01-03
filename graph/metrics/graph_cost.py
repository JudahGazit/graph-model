import abc
import itertools

import math
import numpy as np
from scipy.spatial.distance import euclidean

from graph.distances import perimeter_distance
from graph.graph_dataset import GraphDataset
from graph.metrics.Metric import MetricBoundaries
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.metrics.costs.volume_cost import VolumeCost

costs_mapping = dict(
    resistance=ResistanceCost,
    volume=VolumeCost
)


class GraphCost(abc.ABC):
    def __init__(self, num_nodes, factors: dict, method: str):
        self.num_nodes = num_nodes
        self.factors = factors
        self.cost_boundaries = {key: MetricBoundaries() for key in factors}
        self.method = method
        self.distance_matrix = self._create_distance_matrix()

    def distance(self, i, j):
        raise NotImplementedError()

    def position(self, i):
        raise NotImplementedError()

    def _create_distance_matrix(self):
        mat = np.mat([[self.distance(i, j) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __calculate_total_cost(self, matrix):
        total_cost = 0
        graph_dataset = GraphDataset(distances=self.distance_matrix, positions=self.position, adjacency=matrix)
        for cost_name in self.factors:
            cost = costs_mapping[cost_name](graph_dataset, self.cost_boundaries[cost_name])
            self.cost_boundaries[cost_name] = cost.boundaries
            total_cost += self.factors[cost_name] * cost.cost().normalized_value if self.factors[cost_name] is not None else 0
        return total_cost

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
    def position(self, i):
        phi = 2 * math.pi * i / self.num_nodes
        return [math.cos(phi), math.sin(phi)]

    def distance(self, i, j):
        return perimeter_distance(i, j, self.num_nodes)


class GraphCostLattice(GraphCost):
    def position(self, i):
        n = int(math.sqrt(self.num_nodes))
        return [i % n, int(i / n)]

    def distance(self, i, j):
        return euclidean(self.position(i), self.position(j))


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
    def position(self, i):
        n = int(math.sqrt(self.num_nodes))
        return i % n, int(i / n)

    def distance(self, i, j):
        n = int(math.sqrt(self.num_nodes))
        i_location = self.position(i)
        j_location = self.position(j)
        x = min([abs(i_location[0] - j_location[0]), n - abs(i_location[0] - j_location[0])])
        y = min([abs(i_location[1] - j_location[1]), n - abs(i_location[1] - j_location[1])])
        return np.sqrt(x ** 2 + y ** 2)


class GraphCostFacade:
    type_mapping = {'circular': GraphCostCircular, 'lattice': GraphCostLattice,
                    'sphere': GraphCostSphere, 'torus': GraphCostTorus}

    def get_cost(self, num_nodes, factors, method, type):
        cost_class = self.type_mapping[type]
        return cost_class(num_nodes, factors, method)


if __name__ == '__main__':
    tr = GraphCostTorus(100, 1, 1, 1, 'maximize')
