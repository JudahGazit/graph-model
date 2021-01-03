from typing import Callable, Tuple

import igraph
import networkx as nx
import numpy as np


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
        self.__number_of_nodes_and_edges(graph, adjacency)
        self.__graph_and_adjacency(graph, adjacency)

    def __number_of_nodes_and_edges(self, graph, adjacency):
        if graph is not None:
            self.number_of_nodes = graph.number_of_nodes()
            self.number_of_edges = graph.number_of_edges()
        else:
            self.number_of_nodes = adjacency.shape[0]
            self.number_of_edges = int(np.count_nonzero(adjacency) / 2)

    def __graph_and_adjacency(self, graph, adjacency):
        self.nx_graph = graph
        if adjacency is not None:
            self.adjacency = adjacency
            self.graph = igraph.Graph.Weighted_Adjacency(adjacency.tolist(), mode=igraph.ADJ_UNDIRECTED) \
                if self.number_of_nodes <= 500 else None
        else:
            self.adjacency = nx.to_numpy_matrix(graph)
            self.graph = igraph.Graph.from_networkx(graph) if self.number_of_nodes <= 500 else None