from typing import Callable

import networkx as nx
from geopy.distance import geodesic

from graph.graph_categories.graph_categories import GraphMLGraphCategory
from graph.graph_dataset import GraphDataset


class StreetNetwork(GraphMLGraphCategory):
    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        def distance(u, v):
            u = graph.nodes[u]['name'][1:-1].split(', ')[::-1]
            v = graph.nodes[v]['name'][1:-1].split(', ')[::-1]
            return geodesic(u, v).kilometers
        return distance