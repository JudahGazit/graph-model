import igraph
import numpy as np

from graph.metrics.Metric import MetricBoundaries, Metric
from graph.metrics.costs.icost import ICost


class VolumeCost(ICost):
    def cost(self):
        if self.graph_dataset.widths is not None:
            adjacency = np.multiply(self.graph_dataset.widths, self.graph_dataset.distances)
        else:
            adjacency = self.graph_dataset.adjacency
        if self.graph_dataset.is_connected:
            return Metric(adjacency.sum() / 2, self.boundaries)
        return Metric(float('inf'), self.boundaries)

    def __spanning_tree(self, distance_factor):
        full_graph = igraph.Graph.Weighted_Adjacency((distance_factor * self.graph_dataset.distances).tolist(), igraph.ADJ_UNDIRECTED)
        base_edges = full_graph.spanning_tree('weight', True).get_edgelist()
        edges = np.zeros([self.graph_dataset.number_of_nodes, self.graph_dataset.number_of_nodes])
        edges[tuple(zip(*base_edges))] = 1
        edges = edges + edges.transpose()
        edges = np.multiply(edges, self.graph_dataset.distances)
        return edges

    def __number_of_edges(self):
        if self.graph_dataset.widths is None:
            return self.graph_dataset.number_of_edges
        else:
            return self.graph_dataset.widths.sum() / 2

    def __worst_case(self):
        num_edges = self.__number_of_edges()
        maximal_spanning_tree = self.__spanning_tree(-1)
        worst_case = maximal_spanning_tree.sum() / 2
        edges_in_tree = self.graph_dataset.number_of_nodes - 1
        worst_case += self.graph_dataset.distances.max() * (num_edges - edges_in_tree)
        return worst_case

    def __best_case(self):
        num_edges = self.__number_of_edges()
        minimal_spanning_tree = self.__spanning_tree(1)
        best_case = minimal_spanning_tree.sum() / 2
        edges_in_tree = self.graph_dataset.number_of_nodes - 1
        best_case += self.graph_dataset.distances.min() * (num_edges - edges_in_tree)
        return best_case

    @property
    def boundaries(self) -> MetricBoundaries:
        if self._boundaries.worst_value is None:
            self._boundaries.worst_value = self.__worst_case()
        if self._boundaries.optimal_value is None:
            self._boundaries.optimal_value = self.__best_case()
        return self._boundaries
        # return MetricBoundaries()