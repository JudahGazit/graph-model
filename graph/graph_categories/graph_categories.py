import os
from abc import ABC
from typing import Callable

import pandas as pd
import networkx as nx


class GraphDataset:
    def __init__(self, graph: nx.Graph, distances=None, positions=None):
        self.graph = graph
        self.distances = distances
        self.positions = positions


class GraphCategoryBase(ABC):
    override_weights = True

    def __init__(self, label: str, location: str):
        self.label = label
        self.location = location

    def _load_graph(self, dataset_name: str) -> nx.Graph:
        raise NotImplementedError()

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        return lambda u, v: -1

    def load(self, dataset_name: str) -> GraphDataset:
        graph = nx.nx.convert_node_labels_to_integers(self._load_graph(dataset_name), ordering='sorted', label_attribute='original_label')
        distances = self._load_distance(dataset_name, graph)
        if self.override_weights:
            for edge in graph.edges:
                graph.edges[edge]['weight'] = distances(*edge)
        return GraphDataset(graph, distances)

    def _filter_dir(self, filename):
        return True

    @property
    def options(self):
        if os.path.isdir(self.location):
            return [dataset.rsplit('.', 1)[0] for dataset in os.listdir(self.location) if self._filter_dir(dataset)]
        return []


class CsvGraphCategory(GraphCategoryBase):
    def __init__(self, label, location, src, dst, weight, override_weights=True):
        super().__init__(label, location)
        self.override_weights = override_weights
        self.src = src
        self.dst = dst
        self.weight = weight

    def _load_graph(self, dataset_name):
        df = pd.read_csv(f'{self.location}/{dataset_name}.csv')[[self.src, self.dst, self.weight]]
        df.columns = ['src', 'dst', 'weight']
        graph = nx.Graph()
        graph.add_weighted_edges_from(df.values)
        return graph


class GraphMLGraphCategory(GraphCategoryBase):
    def _load_graph(self, dataset_name):
        graph = nx.read_graphml(f'{self.location}/{dataset_name}.graphml')
        return graph
