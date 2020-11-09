import math

import networkx as nx
from scipy.spatial.distance import euclidean

from graph.graph_categories.graph_categories import GraphDataset


class RandomLattice:
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def distance(self, i, j):
        n = int(math.sqrt(self.num_nodes))
        i_location = i % n, int(i / n)
        j_location = j % n, int(j / n)
        return euclidean(i_location, j_location)

    def randomize(self):
        graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
        for edge in graph.edges:
            graph.edges[edge]['weight'] = self.distance(*edge)
        return GraphDataset(graph, self.distance)