from typing import Callable
import networkx as nx
from geopy.distance import geodesic
import pandas as pd

from graph.datasets.graph_categories import CsvGraphCategory


class HighVoltageCategory(CsvGraphCategory):
    def __init__(self, label, location, src='v_id_1', dst='v_id_2', weight='length_m'):
        super().__init__(label, location, src, dst, weight)

    def _filter_dir(self, filename):
        return 'nodes' not in filename

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        nodes = pd.read_csv(f'{self.location}/{dataset_name}_nodes.csv').set_index('v_id')

        def dist(u, v):
            u = graph.nodes[u]['original_label']
            v = graph.nodes[v]['original_label']
            u = nodes.loc[u][['lat', 'lon']]
            v = nodes.loc[v][['lat', 'lon']]
            return geodesic(u.values, v.values).kilometers

        return dist
