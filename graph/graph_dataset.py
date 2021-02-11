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
        self.__nx_graph(graph)

    def __number_of_nodes_and_edges(self, graph: nx.Graph, adjacency: np.mat):
        self.number_of_nodes = graph.number_of_nodes() if graph is not None else adjacency.shape[0]
        adjacency = adjacency if adjacency is not None else nx.adjacency_matrix(graph, nodelist=range(self.number_of_nodes)).todense()
        self.adjacency = adjacency.astype(np.float)
        if self.widths is None:
            self.widths = np.divide(self.adjacency, self.distances,
                                    out=np.zeros_like(self.adjacency), where=self.distances > 0)
        self.number_of_edges = int(np.nansum(self.widths) // 2)

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

    def __nx_graph(self, graph: nx.Graph):
        self.nx_graph = graph
        self.__add_width_to_nx_graph()

    def __add_width_to_nx_graph(self):
        if self.nx_graph is not None and self.widths is not None:
            for u, v in self.nx_graph.edges:
                if 'width' not in self.nx_graph.edges[u, v]:
                    self.nx_graph.edges[u, v]['width'] = self.widths[u, v]
                else:
                    self.widths[u, v] = self.widths[v, u] = self.nx_graph.edges[u, v]['width']