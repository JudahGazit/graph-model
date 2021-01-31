import json
import logging
import os
import threading

from graph.graph_categories.brain_nets import BrainNet, BrainNetOLD
from graph.graph_categories.graph_categories import CsvGraphCategory
from graph.graph_categories.high_voltage import HighVoltageCategory
from graph.graph_categories.internet import Internet
from graph.graph_categories.roads import Roads
from graph.graph_categories.street_network import StreetNetwork
from graph.graph_formatter import GraphFormatter

logger = logging.getLogger('dataset_loader')

DATASETS = {
    # 'highvoltage': HighVoltageCategory('High Voltage', 'datasets/highvoltage'),
    'street_network': StreetNetwork('Street Network', 'datasets/street_networks', 0.5, 0.05, 0.07),
    'streets_osm': StreetNetwork('Street Network OSM', 'datasets/streets_osm', None, None, 0.015),
    # # 'internet': Internet('Internet', 'datasets/internet'),
    # 'railroads': CsvGraphCategory('Rail Roads', 'datasets/railroads',
    #                               src='from_id',
    #                               dst='to_id',
    #                               weight='weight',
    #                               override_weights=False),
    'brain_nets_old': BrainNetOLD('Brain Nets OLD', 'datasets/brain_nets_old'),
    'brain_nets': BrainNet('Brain Nets', 'datasets/brain_nets'),
    # 'roads': Roads('Roads Network', 'datasets/roads')
}


def list_dir(dir_name):
    if os.path.isdir(dir_name):
        return os.listdir(dir_name)
    return []


class DatasetsLoader:
    def load(self, dataset_name):
        category, dataset_name = dataset_name.split('/', 1)
        graph_category = DATASETS[category]
        dataset = graph_category.load(dataset_name)
        logger.info('loaded dataset %s', dataset_name)
        return dataset

    @property
    def categories(self):
        return DATASETS


class DatasetsResultCache:
    __datasets_loader = DatasetsLoader()
    __lock = threading.Lock()

    def get_from_cache(self, dataset_name):
        if os.path.isfile(f'result_cache/{dataset_name}.json'):
            with open(f'result_cache/{dataset_name}.json') as f:
                result = json.loads(f.read())
                return result

    def write_to_cache(self, dataset_name, payload):
        if not os.path.isfile(f'result_cache/{dataset_name}.json'):
            with open(f'result_cache/{dataset_name}.json', 'w') as f:
                f.write(json.dumps(payload))

    def get_results(self, dataset_name):
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
