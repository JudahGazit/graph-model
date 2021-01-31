import random
from typing import Callable

import networkx as nx
import numpy as np

from graph.graph_categories.graph_categories import GraphMLGraphCategory


class StreetNetwork(GraphMLGraphCategory):
    def __init__(self, label: str, location: str, center_point, center_radius, min_distance):
        super().__init__(label, location)
        self.min_distance = min_distance
        self.center_radius = center_radius
        self.center_point = center_point

    def _load_graph(self, dataset_name):
        graph = super()._load_graph(dataset_name)
        if self.center_point:
            graph = self._load_section_of_graph(graph)
        graph = self._merge_nearby_nodes(graph)
        return graph

    def _merge_nearby_nodes(self, graph):
        nodes = np.array(graph.nodes)
        positions = self._load_positions(graph, nodes)
        distances = self._distance_matrix(positions)
        merge = np.argwhere((distances < self.min_distance) & (distances > 0))
        merge = merge[merge[:, 0] < merge[:, 1]]
        for component in list(nx.connected_components(nx.Graph(merge.tolist()))):
            component = list(component)
            mean_position = positions[component].mean(0)
            keep_node = component[0]
            for node in component[1:]:
                graph = nx.contracted_nodes(graph, nodes[keep_node], nodes[node], self_loops=False)
            graph.nodes[nodes[keep_node]]['X'] = mean_position[0]
            graph.nodes[nodes[keep_node]]['Y'] = mean_position[1]
        largest_cc = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_cc)

    def _load_section_of_graph(self, graph):
        nodes = np.array(graph.nodes)
        positions = self._load_positions(graph, nodes)
        positions = (positions - positions.min(0)) / (positions.max(0) - positions.min(0))
        subset = np.where(
            (abs(positions[:, 0] - self.center_point) < self.center_radius) &
            (abs(positions[:, 1] - self.center_point) < self.center_radius))
        graph = graph.subgraph(nodes[subset])
        return graph

    def _load_positions(self, graph, nodes):
        return np.array([(graph.nodes[i]['X'], graph.nodes[i]['Y']) for i in nodes])

    def _distance_matrix(self, positions):
        x_distance = np.square(np.array(np.meshgrid(positions[:, 0], -positions[:, 0])).sum(0))
        y_distance = np.square(np.array(np.meshgrid(positions[:, 1], -positions[:, 1])).sum(0))
        return np.sqrt(x_distance + y_distance)

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        positions = self._load_positions(graph, range(graph.number_of_nodes()))
        distances = self._distance_matrix(positions)

        return distances.item