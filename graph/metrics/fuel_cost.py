import itertools

import igraph
import networkx as nx
import numpy as np
import pandas as pd

from graph.metrics.Metric import Metric, MetricBoundaries
from graph.metrics.icost import ICost


class FuelCost(ICost):
    def __create_pseudo_manhatten_distances(self, distance_matrix):
        mlattice = np.argsort(distance_matrix, axis=1)[:, 1:3]
        edges = np.zeros([mlattice.shape[0], mlattice.shape[0]])
        for i in range(mlattice.shape[0]):
            for j in range(mlattice.shape[1]):
                target = mlattice[i, j]
                d = self.distances[i, target]
                edges[i, target] = d
                edges[target, i] = d
        lattice = igraph.Graph.Weighted_Adjacency(edges.tolist(), igraph.ADJ_UNDIRECTED)
        manhatten_distances = np.mat(lattice.shortest_paths(weights='weight'))
        return manhatten_distances

    def __create_dataframe_of_distances(self, distance_matrix, manhatten_matrix):
        distances = pd.DataFrame([(i, j, distance_matrix[i, j], manhatten_matrix[i, j])
                                  for i, j in itertools.combinations(range(self.number_of_nodes), 2)]
                                 , columns=['source', 'target', 'l2', 'l1'])
        distances['r'] = distances['l1'] / distances['l2']
        return distances

    def __optimal_fuel_cost(self):
        manhatten_distances = self.__create_pseudo_manhatten_distances(self.distances)
        distances = self.__create_dataframe_of_distances(self.distances, manhatten_distances)
        edges = pd.concat([distances[distances.l2 == 1],
                           distances[distances.r > 1].sort_values('r', ascending=False)])
        edges = edges.head(self.number_of_edges)[['source', 'target', 'l2']].rename({'l2': 'weight'}, axis=1)
        spath = np.mat(igraph.Graph.DataFrame(edges, directed=False).shortest_paths(weights='weight'))
        mean_path = spath[np.triu_indices_from(spath, 1)].mean()
        return mean_path

    def __add_best_loop_edge(self, adjacency):
        source, target = self.__most_useless_edge(adjacency)
        adjacency[source, target] = self.distances[source, target]
        adjacency[target, source] = self.distances[target, source]

    def __most_useless_edge(self, adjacency):
        graph = igraph.Graph.Weighted_Adjacency(adjacency.tolist(), igraph.ADJ_UNDIRECTED)
        shortest_path = np.mat(graph.shortest_paths(weights='weight'))
        data = shortest_path - self.distances
        data[adjacency != 0] = np.nan
        np.fill_diagonal(data, np.nan)
        source, target = np.unravel_index(np.nanargmin(data), data.shape)
        return source, target

    def __create_maximal_spanning_tree(self):
        full_graph = igraph.Graph.Weighted_Adjacency((-self.distances).tolist(), igraph.ADJ_UNDIRECTED)
        base_edges = full_graph.spanning_tree('weight', True).get_edgelist()
        edges = np.zeros([self.number_of_nodes, self.number_of_nodes])
        edges[tuple(zip(*base_edges))] = 1
        edges = edges + edges.transpose()
        edges = np.multiply(edges, self.distances)
        return edges

    def __worst_fuel_cost(self):
        edges = self.__create_maximal_spanning_tree()
        for i in range(self.number_of_edges - self.number_of_nodes + 1):
            self.__add_best_loop_edge(edges)
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