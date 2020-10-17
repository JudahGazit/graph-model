import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset


def __group_by_matrix(df):
    gb = pd.DataFrame(df.flatten())
    gb.columns = ['dist']
    gb['count'] = 0
    gb = gb[gb['dist'] < float('inf')]
    gb = gb.groupby(['dist'], as_index=False).agg({'count': 'count'})
    return gb

def get_penalty_wrapper(regressor, time_of_day_min, time_of_day_max):
    return lambda t, factor: penalty_predict(regressor, t, time_of_day_min, time_of_day_max, factor)

class GraphOptimizer:
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.wiring_factor = wiring_factor
        self.routing_factor = routing_factor
        self.fuel_factor = fuel_factor
        self.distance_matrix = self._create_distance_matrix()

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
        return total_cost

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

    def optimize(self):
        total_possible_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)
        min_value, min_arg = None, None
        initial_edges = random.sample(range(total_possible_edges), self.num_edges)
        for iteration in range(1000):
            initial_edges_vec = [1 if i in initial_edges else 0 for i in range(total_possible_edges)]
            mat = self._triangular_to_mat(initial_edges_vec)
            value = self._target_func(mat)
            if min_value is None or value < min_value:
                min_value = value
                min_arg = mat
        result = np.multiply(self.distance_matrix, min_arg)
        graph = nx.from_numpy_matrix(result)
        return GraphDataset(graph, lambda u, v: perimeter_distance(u, v, self.num_nodes))
