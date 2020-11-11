import os
import re
import numpy as np
import pandas as pd
import json
from glob import glob
from collections import namedtuple

SimulationOptions = namedtuple('SimulationOptions',
                               ['cost_type', 'nodes', 'edges', 'wiring_factor', 'routing_factor', 'fuel_factor'])


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

    def _best_graph(self, graphs, metrics, wiring_factor, routing_factor, fuel_factor):
        graphs_and_metrics = zip(graphs, metrics)
        best_graph = max(graphs_and_metrics, key=lambda t: cost(t[1], wiring_factor, routing_factor, fuel_factor))[0]
        return best_graph

    @property
    def options(self):
        options = glob('optimize_results/*/*_*/*_*_*')
        options = [re.split('[\\\\_/]', option)[2:] for option in options]
        options = [SimulationOptions(*option) for option in options]
        return options

    def _format_results(self, strategy, metrics, charts, edges, *factors):
        result = {
            'edges': self._best_graph(edges, metrics, *factors)
        }
        if strategy == 'mean':
            result['metric'] = self._union_metrics(metrics)
            result['chart'] = self._union_charts(charts)
        elif strategy == 'best':
            result['metric'] = self._best_metric(metrics, *factors)
            result['chart'] = self._best_chart(charts, metrics, *factors)
        return result

    def load(self, strategy, cost_type, nodes, edges, wiring_factor, routing_factor, fuel_factor):
        files = glob(
            f'optimize_results/{cost_type}/{nodes}_{edges}/{wiring_factor}_{routing_factor}_{fuel_factor}/*.json')
        data = [json.loads(open(filename).read()) for filename in files]
        metrics = [d['metrics'] for d in data]
        charts = [d['chart'] for d in data]
        edges = [d.get('edges', []) for d in data]
        return self._format_results(strategy, metrics, charts, edges, wiring_factor, routing_factor, fuel_factor)


if __name__ == '__main__':
    loader = MultipleOptimizationsLoader()
    print(loader.load(200, 4350, 1, 0, 0))
    print(loader.options)
