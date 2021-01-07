import copy
import random
import signal

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from simanneal import Annealer

from graph.metrics.graph_cost import GraphCost, GraphCostTorus
from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase

signal.signal = lambda *args, **kwargs: None


def create_state(matrix):
    return [
        matrix,
        [(a, b) for a, b in np.argwhere(matrix > 0).tolist() if a < b],
    ]

class _GraphAnnealer(Annealer):
    Tmax = 0.0081
    Tmin = 5.8e-18
    steps = 7000 * 3
    updates = 200

    def __init__(self, graph_cost: GraphCost, matrix, *args, **kwargs):
        super().__init__(create_state(matrix))
        self.graph_cost = graph_cost

    def random_change_edge(self):
        add_edge = tuple(np.random.choice(range(self.graph_cost.num_nodes), 2, False))
        remove_edge_index = np.random.randint(len(self.state[1]))
        remove_edge = self.state[1][remove_edge_index]
        if len({*add_edge, *remove_edge}) > 2:
            self.state[0][add_edge] += 1
            self.state[0][add_edge[::-1]] += 1
            self.state[0][remove_edge] -= 1
            self.state[0][remove_edge[::-1]] -= 1
            if self.state[0][add_edge] == 1:
                self.state[1].append(tuple(sorted(add_edge)))
            if self.state[0][remove_edge] == 0:
                self.state[1].pop(remove_edge_index)

    def random_add_edges(self):
        random_edge = tuple(np.random.choice(range(self.graph_cost.num_nodes), 2, False))
        current_value = self.state[0][random_edge]
        if current_value == 0:
            self.state[0][random_edge] = 1
        else:
            self.state[0][random_edge] += np.random.choice([-1, 1])

    def copy_state(self, state):
        matrix, non_zero_indices = state[0], state[1]
        return [matrix.copy(), copy.copy(non_zero_indices)]

    def energy(self):
        score = self.graph_cost.cost(self.state[0])
        return score

    def move(self):
        self.random_change_edge()
        # self.random_add_edges()

def random_gaussian(num_nodes):
    nodes = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_nodes)
    distance_mat = np.mat([[1 / euclidean(u, v) if (u != v).all() else 0 for v in nodes] for u in nodes])
    return distance_mat


def load_distance_mat_of_brain(nodes_num):
    node_distances = pd.read_csv(f'datasets/brain_nets_inter/Cat3_FinalFinal2_interpellated.csv')
    node_distances = node_distances.set_index(['Source', 'Target'])

    def distance(u, v):
        if u != v:
            return node_distances.loc[(u, v)]['Weight']
        return 0

    return np.mat([[distance(i, j) for j in range(nodes_num)] for i in range(nodes_num)])


class SimulatedAnnealing(GraphOptimizerBase):
    def _optimal_matrix(self):
        initial_edges_vec = self.randomize_edges()
        initial_state = self.graph_cost.triangular_to_mat(initial_edges_vec)
        annealer = _GraphAnnealer(self.graph_cost, initial_state)
        # print(annealer.auto(1))
        res, cost = annealer.anneal()
        return res[0]
