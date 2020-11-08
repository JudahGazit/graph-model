import os
import re
import numpy as np
import pandas as pd
import json
from glob import glob
from collections import namedtuple

SimulationOptions = namedtuple('SimulationOptions',
                               ['cost_type', 'nodes', 'edges', 'wiring_factor', 'routing_factor', 'fuel_factor'])


class MultipleOptimizationsLoader:
    def _union_metrics(self, metrics):
        unioned_metrics = {}
        for metric in metrics[0]:
            metric_df = pd.DataFrame([m[metric] for m in metrics])
            unioned_metrics[metric] = metric_df.mean().to_dict()
        return unioned_metrics

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

    @property
    def options(self):
        options = glob('optimize_results/*/*_*/*_*_*')
        options = [re.split('[\\\\_/]', option)[2:] for option in options]
        options = [SimulationOptions(*option) for option in options]
        return options

    def load(self, cost_type, nodes, edges, wiring_factor, routing_factor, fuel_factor):
        files = glob(f'optimize_results/{cost_type}/{nodes}_{edges}/{wiring_factor}_{routing_factor}_{fuel_factor}/*.json')
        data = [json.loads(open(filename).read()) for filename in files]
        metrics = [d['metrics'] for d in data]
        charts = [d['chart'] for d in data]
        return {
            'metric': self._union_metrics(metrics),
            'chart': self._union_charts(charts)
        }


if __name__ == '__main__':
    loader = MultipleOptimizationsLoader()
    print(loader.load(200, 4350, 1, 0, 0))
    print(loader.options)
