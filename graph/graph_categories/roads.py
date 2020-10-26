from typing import Callable

import networkx as nx
import scipy.spatial

from graph.graph_categories.graph_categories import GraphMLGraphCategory


class Roads(GraphMLGraphCategory):
    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        def distance(u, v):
            u = (graph.nodes[u]['X'], graph.nodes[u]['Y'])
            v = (graph.nodes[v]['X'], graph.nodes[v]['Y'])
            return scipy.spatial.distance.euclidean(u, v)
        return distance