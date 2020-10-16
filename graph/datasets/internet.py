from typing import Callable

import networkx as nx
from geopy.distance import geodesic

from graph.datasets.graph_categories import GraphMLGraphCategory


class Internet(GraphMLGraphCategory):
    override_weights = False

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        def distance(u, v):
            u = graph.nodes[u]
            u = (u['lat'], u['lon'])
            v = graph.nodes[v]
            v = (v['lat'], v['lon'])
            return geodesic(u, v).kilometers

        return distance
