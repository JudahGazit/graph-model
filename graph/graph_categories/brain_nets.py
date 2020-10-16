from typing import Callable
import pandas as pd
import networkx as nx

from graph.graph_categories.graph_categories import CsvGraphCategory


class BrainNet(CsvGraphCategory):
    override_weights = False

    def __init__(self, label, location, src='Source', dst='Target', weight='Weight'):
        super().__init__(label, location, src, dst, weight)

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        node_distances = pd.read_csv(f'{self.location}_inter/{dataset_name}_interpellated.csv')
        node_distances = node_distances.set_index([self.src, self.dst])

        def distance(u, v):
            u = graph.nodes[u]['original_label']
            v = graph.nodes[v]['original_label']
            return node_distances.loc[(int(u), int(v))][self.weight]

        return distance
