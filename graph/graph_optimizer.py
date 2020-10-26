import json
import math
import os
import random
from datetime import datetime

from pathos.multiprocessing import ProcessPool
from simanneal import Annealer
import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse
import scipy.special
import signal

from graph.distances import perimeter_distance
from graph.graph_categories.graph_categories import GraphDataset
from graph.graph_formatter import GraphFormatter

POOL_SIZE = 4

signal.signal = lambda *args, **kwargs: None

def stirling_approx_ln_choose(n, k):
    res = k * math.log(n * math.e / k)
    res += -0.5 * math.log(2 * math.pi * k)
    res += - (k ** 2) / (2 * n)
    return res


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

class GraphAnnealer(Annealer):
    copy_strategy = 'slice'
    Tmax = 30
    Tmin = 1

    def __init__(self, graph_cost: GraphCost, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)
        self.graph_cost = graph_cost
        self.len_state = len(state)
        self.state = self.graph_cost.triangular_to_mat(state)

    def move(self):
        a = random.randint(0, self.len_state - 1)
        b = random.randint(0, self.len_state - 1)
        a_index = self.graph_cost.triangular_index(a)
        b_index = self.graph_cost.triangular_index(b)
        a_value, b_value = self.state[a_index], self.state[b_index]
        self.state[a_index[0], a_index[1]] = b_value
        self.state[a_index[1], a_index[0]] = b_value
        self.state[b_index[0], b_index[1]] = a_value
        self.state[b_index[1], b_index[0]] = a_value

    def energy(self):
        return self.graph_cost.cost(self.state)


class GraphOptimizer:
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method='minimize'):
        self.graph_cost = GraphCost(num_nodes, wiring_factor, routing_factor, fuel_factor, method)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.optimize_method = self.minimize_by_annealer

    def minimize_by_random(self, total_possible_edges):
        iterations = 100 * stirling_approx_ln_choose(total_possible_edges,
                                                     self.num_edges)  # Stirling Approx. of ln(n choose k) for n >> k
        iterations = max([int(iterations), 10000])
        iterations = min([iterations, 15000])
        min_value, min_arg = None, None
        for iteration in range(iterations):
            initial_edges = random.sample(range(total_possible_edges), self.num_edges)
            initial_edges_vec = [1 if i in initial_edges else 0 for i in range(total_possible_edges)]
            mat = self.graph_cost.triangular_to_mat(initial_edges_vec)
            value = self.graph_cost.cost(mat)
            if min_value is None or value < min_value:
                min_value = value
                min_arg = mat
        return min_arg

    def minimize_by_annealer(self, total_possible_edges):
        initial_edges = random.sample(range(total_possible_edges), self.num_edges)
        initial_edges_vec = [1 if i in initial_edges else 0 for i in range(total_possible_edges)]
        annealer = GraphAnnealer(self.graph_cost, initial_edges_vec)
        auto_schedule = annealer.auto(1)
        annealer.set_schedule(auto_schedule)
        res, cost = annealer.anneal()
        return res

    def optimize(self):
        total_possible_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)
        min_arg = self.optimize_method(total_possible_edges)
        result = np.multiply(self.graph_cost.distance_matrix, min_arg)
        graph = nx.from_numpy_matrix(result)
        return GraphDataset(graph, lambda u, v: perimeter_distance(u, v, self.num_nodes))

if __name__ == '__main__':
    pool = ProcessPool(8)

    def optimize(i):
        go = GraphOptimizer(30, 60, 0, 1, 0)
        res = go.optimize()
        metrics = GraphFormatter(res).format_metrics()
        with open(f'optimize_results/metrics_annealing5_{i}.json', 'w') as F:
            F.write(json.dumps(metrics, indent=True))
        return res

    values = pool.map(optimize, range(8))
