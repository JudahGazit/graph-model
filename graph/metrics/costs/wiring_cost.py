import itertools

import numpy as np
import random

from graph.metrics.Metric import Metric
from graph.metrics.costs.icost import ICost


class WiringCost(ICost):
    def __estimate_wiring_bounds(self, is_optimal):
        num_nodes = self.graph_dataset.number_of_nodes
        num_edges = self.graph_dataset.number_of_edges
        total_number_of_possible_edges = num_nodes * (num_nodes - 1) / 2
        percentile = num_edges / total_number_of_possible_edges
        random_nodes = random.sample(range(num_nodes), min([200, num_nodes]))
        distances = sorted([self.graph_dataset.distances[u, v] for u, v in itertools.combinations(random_nodes, 2)],
                           reverse=not is_optimal)
        number_of_random_edges = len(distances)
        sum_of_percentile_samples = sum(distances[:max([int(number_of_random_edges * percentile), 1])])
        return sum_of_percentile_samples * (total_number_of_possible_edges / number_of_random_edges)

    def __exact_wiring_bounds(self, is_optimal):
        distance_matrix = self.graph_dataset.distances
        distances = sorted(np.asarray(distance_matrix[np.triu_indices_from(distance_matrix, 1)])[0],
                           reverse=not is_optimal)
        return sum(distances[:min([self.graph_dataset.number_of_edges, len(distances)])])

    def __wiring_cost_bounds(self, is_optimal=True):
        if self.graph_dataset.number_of_nodes > 200:
            return self.__estimate_wiring_bounds(is_optimal)
        else:
            return self.__exact_wiring_bounds(is_optimal)

    @property
    def boundaries(self):
        if self._boundaries.optimal_value is None:
            self._boundaries.optimal_value = self.__wiring_cost_bounds(True)
        if self._boundaries.worst_value is None:
            self._boundaries.worst_value = self.__wiring_cost_bounds(False)
        return self._boundaries

    def cost(self):
        result = Metric(self.graph_dataset.adjacency.sum() / 2, self.boundaries)
        return result
