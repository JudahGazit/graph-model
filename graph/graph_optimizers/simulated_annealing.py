import copy
import threading
from time import sleep

import numpy as np
import random
import signal

from simanneal import Annealer

from graph.graph_optimizers.graph_cost import GraphCost
from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase

signal.signal = lambda *args, **kwargs: None

def create_state(matrix):
        return [
            matrix,
            [[a, b] for a, b in np.argwhere(matrix == 1).tolist() if a < b],
            [[a, b] for a, b in np.argwhere(matrix == 0).tolist() if a < b],
        ]

class _GraphAnnealer(Annealer):
    Tmax = 1
    Tmin = 1e-15
    steps = 25000
    updates = 200

    def copy_state(self, state):
        matrix, non_zero_indices, zero_indices = state
        return [np.copy(matrix), copy.copy(non_zero_indices), copy.copy(zero_indices)]

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

    def move(self):
        self.random_change_edge()


    def energy(self):
        cost = self.graph_cost.cost(self.state[0])
        return cost


class SimulatedAnnealing(GraphOptimizerBase):
    def _optimal_matrix(self):
        initial_edges_vec = self._randomize_edges()
        initial_state = self.graph_cost.triangular_to_mat(initial_edges_vec)
        annealer = _GraphAnnealer(self.graph_cost, initial_state)
        res, cost = annealer.anneal()
        res = res[0]
        return res
