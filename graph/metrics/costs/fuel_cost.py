import igraph
import numpy as np

from graph.metrics.Metric import Metric, MetricBoundaries
from graph.metrics.costs.icost import ICost


class FuelCost(ICost):
    def __optimal_fuel_cost(self):
        edges = self.__minimal_spanning_tree()
        for i in range(self.number_of_edges - self.number_of_nodes + 1):
            self.__add_best_edge(edges)
        graph = igraph.Graph.Weighted_Adjacency(edges.tolist(), igraph.ADJ_UNDIRECTED)
        spath = np.mat(graph.shortest_paths(weights='weight'))
        mean_path = spath[np.triu_indices_from(spath, 1)].mean()
        return mean_path

    def __choose_edge(self, adjacency, minimal):
        graph = igraph.Graph.Weighted_Adjacency(adjacency.tolist(), igraph.ADJ_UNDIRECTED)
        shortest_path = np.mat(graph.shortest_paths(weights='weight'))
        data = shortest_path - self.distances
        data[adjacency != 0] = np.nan
        np.fill_diagonal(data, np.nan)
        if minimal:
            return np.unravel_index(np.nanargmin(data), data.shape)
        else:
            return np.unravel_index(np.nanargmax(data), data.shape)

    def __add_worst_edge(self, adjacency):
        source, target = self.__choose_edge(adjacency, minimal=True)
        adjacency[source, target] = self.distances[source, target]
        adjacency[target, source] = self.distances[target, source]

    def __add_best_edge(self, adjacency):
        source, target = self.__choose_edge(adjacency, minimal=False)
        adjacency[source, target] = self.distances[source, target]
        adjacency[target, source] = self.distances[target, source]

    def __spanning_tree(self, distance_factor):
        full_graph = igraph.Graph.Weighted_Adjacency((distance_factor * self.distances).tolist(), igraph.ADJ_UNDIRECTED)
        base_edges = full_graph.spanning_tree('weight', True).get_edgelist()
        edges = np.zeros([self.number_of_nodes, self.number_of_nodes])
        edges[tuple(zip(*base_edges))] = 1
        edges = edges + edges.transpose()
        edges = np.multiply(edges, self.distances)
        return edges

    def __minimal_spanning_tree(self):
        return self.__spanning_tree(1)

    def __maximal_spanning_tree(self):
        return self.__spanning_tree(-1)

    def __worst_fuel_cost(self):
        edges = self.__maximal_spanning_tree()
        for i in range(self.number_of_edges - self.number_of_nodes + 1):
            self.__add_worst_edge(edges)
        graph = igraph.Graph.Weighted_Adjacency(edges.tolist(), igraph.ADJ_UNDIRECTED)
        spath = np.mat(graph.shortest_paths(weights='weight'))
        mean_path = spath[np.triu_indices_from(spath, 1)].mean()
        return mean_path

    @property
    def boundaries(self) -> MetricBoundaries:
        if self._boundaries.optimal_value is None:
            self._boundaries.optimal_value = self.__optimal_fuel_cost()
        if self._boundaries.worst_value is None:
            self._boundaries.worst_value = self.__worst_fuel_cost()
        return self._boundaries

    def cost(self):
        df = self.all_path_lengths(True)
        fuel_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = Metric(fuel_cost, self.boundaries)
        return result