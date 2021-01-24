import igraph
import numpy as np

from graph.metrics.Metric import MetricBoundaries, Metric
from graph.metrics.costs.icost import INeighboursCost


class VolumeCost(INeighboursCost):
    bound = 10000
    linear_before_bound = False

    def cost(self):
        adjacency = self.format_adjacency_matrix()
        if self.graph_dataset.is_connected:
            value = adjacency.sum() / 2
            metric = Metric(value, self.boundaries)
            metric.normalized_value = self.__penalty_cost(value)
            return metric
        return Metric(float('inf'), self._boundaries)

    def format_adjacency_matrix(self):
        if self.graph_dataset.widths is not None:
            adjacency = np.multiply(self.graph_dataset.widths, self.graph_dataset.distances)
        else:
            adjacency = self.graph_dataset.adjacency
        return adjacency

    def __penalty_cost(self, value):
        if self.bound is not None:
            penalty = np.maximum(0, value - self.bound) ** 2
            return self.linear_before_bound * value + penalty
        return value

    def __spanning_tree(self, distance_factor):
        full_graph = igraph.Graph.Weighted_Adjacency((distance_factor * self.graph_dataset.distances).tolist(),
                                                     igraph.ADJ_UNDIRECTED)
        base_edges = full_graph.spanning_tree('weight', True).get_edgelist()
        edges = np.zeros([self.graph_dataset.number_of_nodes, self.graph_dataset.number_of_nodes])
        edges[tuple(zip(*base_edges))] = 1
        edges = edges + edges.transpose()
        edges = np.multiply(edges, self.graph_dataset.distances)
        return edges

    def __worst_case(self):
        num_edges = self.graph_dataset.number_of_edges
        maximal_spanning_tree = self.__spanning_tree(-1)
        worst_case = maximal_spanning_tree.sum() / 2
        edges_in_tree = self.graph_dataset.number_of_nodes - 1
        worst_case += self.graph_dataset.distances.max() * (num_edges - edges_in_tree)
        return worst_case

    def __best_case(self):
        num_edges = self.graph_dataset.number_of_edges
        minimal_spanning_tree = self.__spanning_tree(1)
        best_case = minimal_spanning_tree.sum() / 2
        edges_in_tree = self.graph_dataset.number_of_nodes - 1
        best_case += self.graph_dataset.distances.min() * (num_edges - edges_in_tree)
        return best_case

    def costs_if_add(self):
        current_cost = self.cost().value
        costs_if_add = current_cost + self.graph_dataset.distances
        costs_if_add = self.__penalty_cost(costs_if_add)
        return costs_if_add

    def costs_if_remove(self):
        current_cost = self.cost().value
        cost_if_remove = current_cost - self.graph_dataset.distances
        cost_if_remove[self.graph_dataset.distances > self.format_adjacency_matrix()] = np.nan
        cost_if_remove = self.__penalty_cost(cost_if_remove)
        return cost_if_remove

    @property
    def boundaries(self) -> MetricBoundaries:
        return MetricBoundaries()
