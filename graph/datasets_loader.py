import glob
import json
import os
import threading
from multiprocessing.pool import Pool

from graph.graph_categories.brain_nets import BrainNet
from graph.graph_categories.graph_categories import CsvGraphCategory
from graph.graph_categories.high_voltage import HighVoltageCategory
from graph.graph_categories.internet import Internet
from graph.graph_categories.roads import Roads
from graph.graph_categories.street_network import StreetNetwork
from graph.graph_formatter import GraphFormatter

DATASETS = {
    'highvoltage': HighVoltageCategory('High Voltage', 'datasets/highvoltage'),
    'street_network': StreetNetwork('Street Network', 'datasets/street_networks'),
    'internet': Internet('Internet', 'datasets/internet'),
    'railroads': CsvGraphCategory('Rail Roads', 'datasets/railroads',
                                  src='from_id',
                                  dst='to_id',
                                  weight='weight',
                                  override_weights=False),
    'brain_nets': BrainNet('Brain Nets', 'datasets/brain_nets'),
    'roads': Roads('Roads Network', 'datasets/roads')
}


def list_dir(dir_name):
    if os.path.isdir(dir_name):
        return os.listdir(dir_name)
    return []


class DatasetsLoader:
    def load(self, dataset_name):
        category, dataset_name = dataset_name.split('/', 1)
        graph_category = DATASETS[category]
        return graph_category.load(dataset_name)

    @property
    def categories(self):
        return DATASETS


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

    def options(self):
        result = []
        for category in self.__datasets_loader.categories:
            datasets_input = self.__datasets_loader.categories[category].options
            datasets_cache = [dataset.rsplit('.', 1)[0] for dataset in list_dir(f'result_cache/{category}')]
            datasets = sorted(set(datasets_input + datasets_cache))
            result.append({"name": category, "label": DATASETS[category].label,
                           "options": [{"name": dataset, "label": dataset} for dataset in datasets]
                           })
        return result


def every_option(option):
    try:
        drc_local = DatasetsResultCache()
        drc_local.get_results(option)
    except Exception as e:
        print(f'ERROR {option}: {e}')


if __name__ == '__main__':
    # json_files = glob.glob('./result_cache/*/*.json')
    # print(json_files)
    # for json_file in json_files:
    #     print(json_file)
    #     os.remove(json_file)
    graph = DatasetsLoader().load('roads/uk').graph
    print('loaded graph')
    import networkx as nx
    comps = list(nx.connected_components(graph))
    print('calculated comps')
    giant = max(comps, key=len)
    nx.write_graphml(graph.subgraph(giant), 'datasets/roads/uk_mainland.graphml')
    print('created giant')
    drc = DatasetsResultCache()
    # pool = Pool(8)
    # for category in drc.options():
    #     print(category['name'])
    #     m = pool.map(every_option, ['/'.join([category['name'], option['name']]) for option in category['options']])
    #     list(m)
    #
    drc.get_results('roads/uk_mainland')