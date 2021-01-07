import os
import re
import numpy as np
import pandas as pd
import json
from glob import glob
from collections import namedtuple

SimulationOptions = namedtuple('SimulationOptions',
                               ['cost_type', 'nodes', 'edges', 'remark', 'wiring_factor', 'routing_factor', 'fuel_factor'])


def cost(metrics, wiring_factor, routing_factor, fuel_factor):
    wiring = metrics['wiring-cost']['normalized_value']
    routing = metrics['routing-cost']['normalized_value']
    fuel = metrics['fuel-cost']['normalized_value']
    return float(wiring_factor) * wiring + float(routing_factor) * routing + float(fuel_factor) * fuel


class MultipleOptimizationsLoader:
    def _union_metrics(self, metrics):
        unioned_metrics = {}
        for metric in metrics[0]:
            metric_df = pd.DataFrame([m[metric] for m in metrics])
            unioned_metrics[metric] = metric_df.mean().to_dict()
        return unioned_metrics

    def _best_metric(self, metrics, wiring_factor, routing_factor, fuel_factor):
        return max(metrics, key=lambda m: cost(m, wiring_factor, routing_factor, fuel_factor))

    def _union_charts(self, charts):
        unioned_charts = {}
        for chart in charts[0]:
            x = charts[0][chart]['x']
            ys = [c[chart]['y'] + ([0] * (len(x) - len(c[chart]['y']))) for c in charts]
            y_mean = np.mat(list(zip(*ys))).mean(axis=1).flatten().tolist()[0]
            unioned_charts[chart] = {
                'x': x,
                'y': y_mean,
                'ys': ys
            }
        return unioned_charts

    def _best_chart(self, charts, metrics, wiring_factor, routing_factor, fuel_factor):
        charts_and_metrics = zip(charts, metrics)
        best_chart = max(charts_and_metrics, key=lambda t: cost(t[1], wiring_factor, routing_factor, fuel_factor))[0]
        return best_chart

    def _best_graph(self, graphs, widths, metrics, wiring_factor, routing_factor, fuel_factor):
        graphs_and_metrics = zip(widths, graphs, metrics)
        best_widths, best_graph, _ = max(graphs_and_metrics, key=lambda t: cost(t[2], wiring_factor, routing_factor, fuel_factor))
        return best_graph, best_widths

    @property
    def options(self):
        options = glob('optimize_results/*/*_*/*_*_*')
        loaded_options = []
        for option in options:
            cost_type, network_size, factors = option.split('/')[1:]
            network_size = (network_size.split('_', 2) + [''])[:3]
            factors = factors.split('_', 3)
            loaded_options.append(SimulationOptions(cost_type, *network_size, *factors))
        return loaded_options

    def _format_results(self, strategy, metrics, charts, edges, widths, *factors):
        is_floats = all(['*' not in v for v in factors])
        best_graph, best_widths = self._best_graph(edges, widths, metrics, *factors) if is_floats else ([], None)
        result = {
            'edges': best_graph,
            'widths': best_widths
        }
        if strategy == 'mean':
            result['metric'] = self._union_metrics(metrics)
            result['chart'] = self._union_charts(charts)
        elif strategy == 'best' and is_floats:
            result['metric'] = self._best_metric(metrics, *factors)
            result['chart'] = self._best_chart(charts, metrics, *factors)
        return result

    def _load_files(self, file_names):
        result = []
        for filename in file_names:
            try:
                result.append(json.loads(open(filename).read()))
            except:
                raise RuntimeError(f'falied to read `{filename}`')
        return result

    def load(self, strategy, cost_type, nodes, edges, remark, wiring_factor, routing_factor, fuel_factor):
        remark = "_" + remark if len(remark) else remark
        files = glob(
            f'optimize_results/{cost_type}/{nodes}_{edges}{remark}/{wiring_factor}_{routing_factor}_{fuel_factor}/*.json')
        data = self._load_files(files)
        metrics = [d['metrics'] for d in data]
        charts = [d['chart'] for d in data]
        edges = [d.get('edges', []) for d in data]
        widths = [d.get('widths', None) for d in data]
        return self._format_results(strategy, metrics, charts, edges, widths, wiring_factor, routing_factor, fuel_factor)


if __name__ == '__main__':
    loader = MultipleOptimizationsLoader()
    print(loader.load(200, 4350, 1, 0, 0))
    print(loader.options)
