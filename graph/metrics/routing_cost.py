import igraph
import numpy as np

from graph.metrics.Metric import Metric
from graph.metrics.icost import ICost


class RoutingCost(ICost):
    def __optimal_routing_cost(self):
        number_of_pairs = self.number_of_nodes * (self.number_of_nodes - 1) / 2
        number_of_pairs_in_distance_1 = self.number_of_edges
        number_of_pairs_in_distance_2 = number_of_pairs - self.number_of_edges
        mean_degree = (number_of_pairs_in_distance_1 + 2 * number_of_pairs_in_distance_2) / number_of_pairs
        return mean_degree

    def __add_loop_edge(self, adjacency, last_insert):
        available_edges = np.where(adjacency[:last_insert, last_insert] == 0)
        if len(available_edges[0]) > 0:
            i = int(available_edges[0][0])
            adjacency[i, last_insert] = adjacency[last_insert, i] = 1
            return last_insert
        else:
            available_inserts = np.where(adjacency[0, 1:] == 0)
            i = int(available_inserts[0][0]) + 1
            adjacency[0, i] = adjacency[i, 0] = 1
            return i

    def __worst_routing_cost(self):
        adjacency = self.__create_chain_graph()
        last_insert = 0
        for iteration in range(self.number_of_edges - self.number_of_nodes + 1):
            last_insert = self.__add_loop_edge(adjacency, last_insert)
        mean_path = self.__mean_shortest_path(adjacency)
        return mean_path

    def __create_chain_graph(self):
        edges = np.zeros([self.number_of_nodes, self.number_of_nodes])
        for i in range(self.number_of_nodes - 1):
            edges[i, i + 1] = 1
            edges[i + 1, i] = 1
        return edges

    def __mean_shortest_path(self, adjacency: np.mat):
        g = igraph.Graph.Adjacency(adjacency.tolist(), igraph.ADJ_UNDIRECTED)
        shortest_paths = np.mat(g.shortest_paths())
        mean_path = shortest_paths[np.triu_indices_from(shortest_paths, 1)].mean()
        return mean_path

    def cost(self):
        df = self.all_path_lengths(False)
        routing_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = Metric(routing_cost, self.boundaries)
        return result

    @property
    def boundaries(self):
        if self._boundaries.optimal_value is None:
            self._boundaries.optimal_value = self.__optimal_routing_cost()
        if self._boundaries.worst_value is None:
            self._boundaries.worst_value = self.__worst_routing_cost()
        return self._boundaries