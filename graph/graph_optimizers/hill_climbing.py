import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import softmax

from graph.graph_dataset import GraphDataset
from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.metrics.graph_cost import costs_mapping


class HillClimbing(GraphOptimizerBase):
    epochs = 500
    eps = 20

    def _optimal_matrix(self):
        # initial_edges_vec = self.randomize_edges()
        # state = self.graph_cost.triangular_to_mat(initial_edges_vec)
        state = np.zeros_like(self.graph_cost.distance_matrix)
        state[self.graph_cost.distance_matrix == 1] = 1
        current_cost = None
        minimum = None
        minimum_state = None
        minimum_steady = 0
        for i in range(self.epochs):
            if minimum_steady >= self.eps:
                state = minimum_state.copy()
                current_cost = minimum
                minimum_steady = 0
            else:
                current_cost = self.graph_cost.cost(state)
            if minimum is None or current_cost < minimum:
                minimum = current_cost
                minimum_state = state.copy()
                minimum_steady = 0
            else:
                minimum_steady += 1
            neighbours = self.neighbours_costs(state)
            src, dst, is_add = self.move(state, neighbours)
            # if i % (self.epochs // 200 + 1) == 0:
            print(i, src, dst, is_add, minimum, sep='\t\t')
        return minimum_state

    def neighbours_costs(self, state, current_value=None):
        costs_if_add = np.zeros_like(self.graph_cost.distance_matrix)
        costs_if_remove = np.zeros_like(self.graph_cost.distance_matrix)
        state_dataset = GraphDataset(distances=self.graph_cost.distance_matrix, positions=self.graph_cost.position,
                                     adjacency=np.multiply(state, self.graph_cost.distance_matrix))
        for cost_name in costs_mapping:
            cost = costs_mapping[cost_name](state_dataset)
            factor = self.graph_cost.factors[cost_name]
            costs_if_add += cost.costs_if_add() * factor
            costs_if_remove += cost.costs_if_remove() * factor
        costs = np.stack([np.asarray(costs_if_remove), np.asarray(costs_if_add)], axis=2)
        costs = pd.DataFrame(np.column_stack([np.argwhere(~np.isnan(costs)), costs[~np.isnan(costs)]]),
                             columns=['src', 'dst', 'is_add', 'cost'])
        costs = costs[costs['src'] < costs['dst']]
        if current_value is not None:
            costs = costs[costs['cost'] < current_value]
        if len(costs) > 0:
            costs['p'] = softmax(-costs['cost'])
        return costs

    def move(self, state, neighbours, sampled=None):
        # print(sampled)
        # print(neighbours.sort_values('cost').head(5))
        sampled = neighbours.sort_values('cost').head(1)
        # sampled = sampled if sampled is not None else neighbours.sample(1, weights=neighbours['p'])
        src, dst, is_add, cost = [int(sampled[item].values[0]) for item in ['src', 'dst', 'is_add', 'cost']]
        factor = 1 if is_add > 0 else -1
        state[src, dst] += factor
        state[dst, src] += factor
        return src, dst, is_add