import math
import random
from datetime import time, datetime

from pathos.multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse
import scipy.special

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset
import matplotlib.pyplot as plt

POOL_SIZE = 4


def stirling_approx_ln_choose(n, k):
    res = k * math.log(n * math.e / k)
    res += -0.5 * math.log(2 * math.pi * k)
    res += - (k ** 2) / (2 * n)
    return res


class GraphOptimizer:
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.wiring_factor = wiring_factor
        self.routing_factor = routing_factor
        self.fuel_factor = fuel_factor
        self.distance_matrix = self._create_distance_matrix()
        self.method = method

    def _create_distance_matrix(self):
        mat = np.mat([[perimeter_distance(i, j, self.num_nodes) for j in range(self.num_nodes)]
                      for i in range(self.num_nodes)])
        return mat

    def __all_path_lengths(self, matrix, weight=False):
        indices = random.sample(range(self.num_nodes), min([1000, self.num_nodes]))
        all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(matrix, directed=False,
                                                                     unweighted=not weight, indices=indices)
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

    def _target_func(self, mat):
        matrix = np.multiply(self.distance_matrix, mat)
        total_cost = self.__calculate_total_cost(matrix)
        method_factor = 1 if self.method == 'minimize' else -1
        return method_factor * total_cost

    def _constraint(self, mat_as_vector):
        mat = np.triu(mat_as_vector.reshape((self.num_nodes, self.num_nodes)))
        mat = np.round(mat)
        return mat.sum() == self.num_edges

    def _triangular_to_mat(self, triangular_as_vec):
        vec_iter = iter(triangular_as_vec)
        mat = np.mat([[
            next(vec_iter) if j > i else 0 for j in range(self.num_nodes)
        ] for i in range(self.num_nodes)])
        return mat + mat.transpose()

    def minimize_by_random(self, iterations, total_possible_edges):
        min_value, min_arg = None, None
        for iteration in range(iterations):
            initial_edges = random.sample(range(total_possible_edges), self.num_edges)
            initial_edges_vec = [1 if i in initial_edges else 0 for i in range(total_possible_edges)]
            mat = self._triangular_to_mat(initial_edges_vec)
            value = self._target_func(mat)
            if min_value is None or value < min_value:
                min_value = value
                min_arg = mat
        return min_arg

    def optimize(self):
        total_possible_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)
        iterations = 100 * stirling_approx_ln_choose(total_possible_edges,
                                                     self.num_edges)  # Stirling Approx. of ln(n choose k) for n >> k
        print(iterations)
        iterations = max([int(iterations), 10000])
        iterations = min([iterations, 15000])
        pool = Pool(POOL_SIZE)
        local_min_args = pool.map(lambda _:
                                  self.minimize_by_random(int(iterations / POOL_SIZE), total_possible_edges),
                                  range(POOL_SIZE))
        min_arg = min(local_min_args, key=self._target_func)
        result = np.multiply(self.distance_matrix, min_arg)
        graph = nx.from_numpy_matrix(result)
        plt.show()
        return GraphDataset(graph, lambda u, v: perimeter_distance(u, v, self.num_nodes))


if __name__ == '__main__':
    go = GraphOptimizer(50, 100, 1, 10, 0)
    now = datetime.now()
    print(now)
    value = go.optimize()
    print(f'{datetime.now()}, took {(datetime.now() - now).total_seconds()}')
    print(value)
