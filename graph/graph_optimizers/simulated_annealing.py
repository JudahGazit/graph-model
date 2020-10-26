import random
import signal

from simanneal import Annealer

from graph.graph_optimizers.graph_cost import GraphCost
from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase

signal.signal = lambda *args, **kwargs: None


class _GraphAnnealer(Annealer):
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


class SimulatedAnnealing(GraphOptimizerBase):
    def _optimal_matrix(self):
        initial_edges_vec = self._randomize_edges()
        annealer = _GraphAnnealer(self.graph_cost, initial_edges_vec)
        auto_schedule = annealer.auto(1)
        annealer.set_schedule(auto_schedule)
        res, cost = annealer.anneal()
        return res
