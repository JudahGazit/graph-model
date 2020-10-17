import math
import random

import networkx as nx

from graph.distances import perimeter_distance, hierarchical_distance
from graph.graph_categories.graph_categories import GraphDataset


def get_probability(num_leaves, A, B, alpha, beta, a, b):
    result = A * math.exp(- alpha * perimeter_distance(a, b, num_leaves))
    result += B * math.exp(beta * hierarchical_distance(a, b))
    assert result <= 1
    return result


def randomize_in_prob(p):
    return random.random() <= p


class GraphSimulator:
    def __init__(self, num_leaves, A, B, alpha, beta):
        self.num_leaves = num_leaves
        self.A = A
        self.B = B
        self.alpha = alpha
        self.beta = beta

    def simulate(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_leaves))
        for i in range(self.num_leaves):
            for j in range(self.num_leaves):
                if i < j:
                    p = get_probability(self.num_leaves, self.A, self.B, self.alpha, self.beta, i, j)
                    if randomize_in_prob(p):
                        graph.add_edge(i, j, weight=perimeter_distance(i, j, self.num_leaves))
        graph_dataset = GraphDataset(graph, lambda u, v: perimeter_distance(u, v, self.num_leaves))
        return graph_dataset
