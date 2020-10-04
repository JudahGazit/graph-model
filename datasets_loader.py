import json
import os
import threading
from collections import namedtuple

import pandas as pd
import networkx as nx
from scipy.io import loadmat

from graph_formatter import GraphFormatter

dataset_fields = ['location', 'src', 'dst', 'weight', 'format', 'label']
Dataset = namedtuple('Dataset', dataset_fields, defaults=(None, ) * len(dataset_fields))

DATASETS = {
    'highvoltage': Dataset(location='datasets/highvoltage',
                           src='v_id_1',
                           dst='v_id_2',
                           weight='length_m',
                           format='csv',
                           label='High Voltage'
                           ),
    'street_network': Dataset(location='datasets/street_networks',
                               format='graphml',
                               label='Street Network'
                               ),
    'internet': Dataset(location='datasets/internet',
                        format='csv',
                        label='Internet',
                        src='src',
                        dst='dst',
                        weight='weight',
                      ),
    'railroads': Dataset(location='datasets/railroads',
                         format='csv',
                         label='Rail Roads',
                         src='from_id',
                         dst='to_id',
                         weight='weight',
                         ),
}


class DatasetsLoader:
    def __init__(self):
        self.loaders = {
            'csv': self.load_csv,
            'graphml': self.load_graphml
        }

    def load(self, dataset_name):
        category, dataset_name = dataset_name.split('/', 1)
        dataset = DATASETS[category]
        graph = self.loaders[dataset.format](dataset_name, dataset)
        graph = nx.convert_node_labels_to_integers(graph)
        return graph

    def load_csv(self, dataset_name, dataset):
        df = pd.read_csv(f'{dataset.location}/{dataset_name}.csv')[[dataset.src, dataset.dst, dataset.weight]]
        df.columns = ['src', 'dst', 'weight']
        graph = nx.Graph()
        graph.add_weighted_edges_from(df.values)
        return graph

    def load_graphml(self, dataset_name, dataset):
        graph = nx.read_graphml(f'{dataset.location}/{dataset_name}.graphml')
        return graph

    def fetch_options(self):
        result = []
        for category in DATASETS:
            datasets_input = [dataset.rsplit('.', 1)[0] for dataset in os.listdir(DATASETS[category].location)]
            datasets_cache = [dataset.rsplit('.', 1)[0] for dataset in os.listdir(f'result_cache/{category}')]
            datasets = sorted(set(datasets_input + datasets_cache))
            result.append({"name": category, "label": DATASETS[category].label,
                           "options": [{"name": dataset, "label": dataset} for dataset in datasets]
                           })
        return result


class DatasetsResultCache:
    __datasets_loader = DatasetsLoader()
    __result_cache = {}
    __lock = threading.Lock()

    def get_from_cache(self, dataset_name):
        if dataset_name not in self.__result_cache:
            if os.path.isfile(f'result_cache/{dataset_name}.json'):
                with open(f'result_cache/{dataset_name}.json') as f:
                    result = json.loads(f.read())
                self.__result_cache[dataset_name] = result
        return self.__result_cache.get(dataset_name)

    def write_to_cache(self, dataset_name, payload):
        self.__result_cache[dataset_name] = payload
        if not os.path.isfile(f'result_cache/{dataset_name}.json'):
            with open(f'result_cache/{dataset_name}.json', 'w') as f:
                f.write(json.dumps(payload))

    def get_results(self, dataset_name):
        with self.__lock:
            if not self.get_from_cache(dataset_name):
                graph = self.__datasets_loader.load(dataset_name)
                formatter = GraphFormatter(graph)
                result = {
                    'edges': formatter.format_graph_sample(),
                    'chart': formatter.format_chart(),
                    'metric': formatter.format_metrics()
                }
                self.write_to_cache(dataset_name, result)
        return self.get_from_cache(dataset_name)

    def fetch_options(self):
        return self.__datasets_loader.fetch_options()

