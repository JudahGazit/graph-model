import random

import numpy as np
import pandas as pd
import scipy.sparse

from graph.distances import perimeter_distance


class GraphCost:
    def __init__(self, num_nodes, wiring_factor, routing_factor, fuel_factor, method):
        self.num_nodes = num_nodes
        self.wiring_factor = wiring_factor
        self.fuel_factor = fuel_factor
        self.method = method
        self.distance_matrix = self._create_distance_matrix()
        self.routing_factor = routing_factor

    def _create_distance_matrix(self):
        mat = np.mat([[perimeter_distance(i, j, self.num_nodes) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __all_path_lengths(self, matrix, weight=False):
        sample_size = min([1000, self.num_nodes])
        indices = random.sample(range(self.num_nodes), sample_size) if sample_size != self.num_nodes else None
        all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(matrix, directed=False,
                                                                     unweighted=not weight, indices=indices, method='D')
        gb = pd.DataFrame(all_pairs_shortest_path.flatten(), columns=['dist'])
        gb['count'] = 0
        gb = gb[gb['dist'] < float('inf')].groupby(['dist'], as_index=False).agg({'count': 'count'})
        return gb

    def __calculate_total_cost(self, matrix):
        total_cost = 0
        if self.wiring_factor:
            wiring = matrix.sum() / 2
            total_cost += self.wiring_factor * wiring
        if self.routing_factor:
            routing = self.__all_path_lengths(matrix, False)
            routing = (routing['dist'] * routing['count']).sum() / (routing['count']).sum()
            total_cost += self.routing_factor * routing
        if self.fuel_factor:
            fuel = self.__all_path_lengths(matrix, True)
            fuel = (fuel['dist'] * fuel['count']).sum() / (fuel['count']).sum()
            total_cost += self.fuel_factor * fuel
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
        vec_iter = iter(triangular_as_vec)
        mat = np.mat([[
            next(vec_iter) if j > i else 0 for j in range(self.num_nodes)
        ] for i in range(self.num_nodes)])
        return mat + mat.transpose()