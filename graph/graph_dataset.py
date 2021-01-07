from typing import Callable, Tuple

import igraph
import networkx as nx
import numpy as np
import scipy.sparse.csgraph


class GraphDataset:
    def __init__(self,
                 graph: nx.Graph = None,
                 distances: np.mat = None,
                 positions: Callable[[int], Tuple[int, int]] = None,
                 widths: np.mat = None,
                 adjacency: np.mat = None
                 ):
        self.distances = distances
        self.positions = positions
        self.widths = widths
        self.__graph = None
        self.__is_connected = None
        self.__number_of_nodes_and_edges(graph, adjacency)
        self.__graph_and_adjacency(graph, adjacency)

    def __number_of_nodes_and_edges(self, graph, adjacency):
        self.number_of_nodes = graph.number_of_nodes() if graph is not None else adjacency.shape[0]
        if self.widths is not None:
            self.number_of_edges = self.widths.sum() // 2
        else:
            self.number_of_edges = graph.number_of_edges() if graph is not None else np.count_nonzero(adjacency) // 2

    @property
    def is_connected(self):
        if self.__is_connected is None:
            if self.__graph is not None:
                self.__is_connected = self.graph.is_connected()
            elif self.nx_graph is not None:
                self.__is_connected = nx.is_connected(self.nx_graph)
            else:
                self.__is_connected = scipy.sparse.csgraph.connected_components(self.adjacency, False, return_labels=False) == 1
        return self.__is_connected

    @property
    def graph(self):
        if self.__graph is None:
            if self.number_of_nodes <= 500:
                if self.nx_graph is not None:
                    self.__graph = igraph.Graph.from_networkx(self.nx_graph)
                else:
                    self.__graph = igraph.Graph.Weighted_Adjacency(self.adjacency.tolist(), mode=igraph.ADJ_UNDIRECTED)
        return self.__graph

    def __graph_and_adjacency(self, graph, adjacency):
        self.nx_graph = graph
        self.__add_width_to_nx_graph()
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = nx.to_numpy_matrix(graph)

    def __add_width_to_nx_graph(self):
        if self.nx_graph is not None and self.widths is not None:
            for edge in self.nx_graph.edges:
                self.nx_graph.edges[edge]['width'] = self.widths[edge[0], edge[1]]