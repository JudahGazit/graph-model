from collections import namedtuple

import pandas as pd
import networkx as nx

from graph_formatter import GraphFormatter

Dataset = namedtuple('Dataset', ['location', 'src', 'dst', 'weight'])

DATASETS = {
    'europe-highvoltage': Dataset(location='datasets/gridkit_europe-highvoltage-links.csv',
                                  src='v_id_1',
                                  dst='v_id_2',
                                  weight='length_m'),
    'america-highvoltage': Dataset(location='datasets/gridkit_north_america-highvoltage-links.csv',
                                      src='v_id_1',
                                      dst='v_id_2',
                                      weight='length_m'),
}


class DatasetsLoader:
    def load(self, dataset_name):
        dataset = DATASETS[dataset_name]
        df = pd.read_csv(dataset.location)[[dataset.src, dataset.dst, dataset.weight]]
        df.columns = ['src', 'dst', 'weight']
        graph = nx.Graph()
        graph.add_weighted_edges_from(df.values)
        graph = nx.convert_node_labels_to_integers(graph)
        return graph


class DatasetsResultCache:
    __datasets_loader = DatasetsLoader()
    __result_cache = {}

    def get_results(self, dataset_name):
        if dataset_name not in self.__result_cache:
            graph = self.__datasets_loader.load(dataset_name)
            formatter = GraphFormatter(graph)
            result = {
                'edges': formatter.format_graph_sample(),
                'chart': formatter.format_chart(),
                'metric': formatter.format_metrics()
            }
            self.__result_cache[dataset_name] = result
        return self.__result_cache[dataset_name]


if __name__ == '__main__':
    dataset_cache = DatasetsResultCache()
    print(dataset_cache.get_results('america-highvoltage'))