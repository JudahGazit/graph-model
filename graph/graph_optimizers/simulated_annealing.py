import networkx as nx
import math
import sys
import time

import matplotlib.pyplot as plt
import copy
import random
import signal

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from simanneal import Annealer
from simanneal.anneal import time_string

from graph.graph_categories.graph_categories import GraphDataset
from graph.graph_formatter import GraphFormatter
from graph.graph_optimizers.graph_cost import GraphCost
from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase

signal.signal = lambda *args, **kwargs: None


# TODO: cleanup

def create_state(matrix):
    return [
        matrix,
        [[a, b] for a, b in np.argwhere(matrix == 1).tolist() if a < b],
        [[a, b] for a, b in np.argwhere(matrix == 0).tolist() if a < b],
    ]

class _GraphAnnealer(Annealer):
    Tmax = 0.012
    Tmin = 1.1e-08
    steps = 10000
    updates = 200

    def __init__(self, graph_cost: GraphCost, matrix, *args, **kwargs):
        super().__init__(create_state(matrix))
        self.graph_cost = graph_cost

    def random_change_edge(self):
        a_index = random.randrange(len(self.state[1]))
        b_index = random.randrange(len(self.state[2]))
        a_index = self.state[1].pop(a_index)
        b_index = self.state[2].pop(b_index)
        self.state[1].append(b_index), self.state[2].append(a_index)
        self.state[0][a_index[0], a_index[1]] = 0
        self.state[0][a_index[1], a_index[0]] = 0
        self.state[0][b_index[0], b_index[1]] = 1
        self.state[0][b_index[1], b_index[0]] = 1

    def copy_state(self, state):
        matrix, non_zero_indices, zero_indices = state
        return [np.copy(matrix), copy.copy(non_zero_indices), copy.copy(zero_indices)]

    def energy(self):
        # score = self._energy_by_target()
        score = self.graph_cost.cost(self.state[0])
        return score

    def move(self):
        self.random_change_edge()

    def _energy_by_target(self):
        matrix = np.multiply(self.graph_cost.distance_matrix, self.state[0])
        metrics = self.graph_cost.create_graph_metrics(matrix)
        wiring_cost = metrics.wiring_cost().normalized_value
        routing_cost = metrics.routing_cost().normalized_value
        fuel_cost = metrics.fuel_cost().normalized_value
        score = 0
        for cost, goal in zip([wiring_cost, routing_cost, fuel_cost],
                              [self.graph_cost.wiring_factor, self.graph_cost.routing_factor,
                               self.graph_cost.fuel_factor]):
            if goal is not None:
                score += (1000 * (cost - goal)) ** 2
        if score < 1e-1:
            self.user_exit = True
        return score


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
        initial_edges_vec = self._randomize_edges()
        initial_state = self.graph_cost.triangular_to_mat(initial_edges_vec)
        annealer = _GraphAnnealer(self.graph_cost, initial_state)
        res, cost = annealer.anneal()
        res = res[0]
        return res
