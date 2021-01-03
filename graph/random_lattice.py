import math

import networkx as nx
from scipy.spatial.distance import euclidean

from graph.graph_dataset import GraphDataset
from graph.graph_optimizers.graph_cost import GraphCostTorus


class RandomLattice:
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.cost = GraphCostTorus(num_nodes, None, None, None, 'maximize')

    def position(self, i):
        return self.cost.position(i)

    def distance(self, i, j):
        return self.cost.distance(i, j)

    def randomize(self):
        graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
        for edge in graph.edges:
            graph.edges[edge]['weight'] = self.distance(*edge)
        return GraphDataset(graph, self.cost.distance_matrix, self.position)