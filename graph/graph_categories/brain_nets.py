from typing import Callable
import pandas as pd
import networkx as nx
import scipy.spatial


from graph.graph_categories.graph_categories import CsvGraphCategory, GraphMLGraphCategory


class BrainNetOLD(CsvGraphCategory):
    override_weights = False

    def __init__(self, label, location, src='Source', dst='Target', weight='Weight'):
        super().__init__(label, location, src, dst, weight)

    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        node_distances = pd.read_csv(f'{self.location.replace("_old", "_inter")}/{dataset_name}_interpellated.csv')
        node_distances = node_distances.set_index([self.src, self.dst])

        def distance(u, v):
            if u != v:
                u = graph.nodes[u]['original_label']
                v = graph.nodes[v]['original_label']
                return node_distances.loc[(int(u), int(v))][self.weight]
            return 0

        return distance


class BrainNet(GraphMLGraphCategory):
    def _load_distance(self, dataset_name: str, graph: nx.Graph) -> Callable[[int, int], float]:
        def distance(u, v):
            u = (graph.nodes[u]['X'], graph.nodes[u]['Y'], graph.nodes[u]['Z'])
            v = (graph.nodes[v]['X'], graph.nodes[v]['Y'], graph.nodes[v]['Z'])
            return scipy.spatial.distance.euclidean(u, v)
        return distance
