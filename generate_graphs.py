import json
import logging
import math
import os
import random
import sys
from glob import glob
from shutil import rmtree

import numpy as np
import pandas as pd
import skopt
from pathos.multiprocessing import ProcessPool

from graph.datasets_loader import DatasetsResultCache
from graph.graph_formatter import GraphFormatter
from graph.graph_optimizer import GraphOptimizer
from graph.graph_simulator import GraphSimulator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def randomize(i):
    N, p = random.randint(10, 2000), random.random() * (0.5 - 1e-3) + 1e-3
    simulate = GraphSimulator(N, p, 0, 0, 0).simulate()
    metrics = GraphFormatter(simulate).format_metrics()
    degree = pd.DataFrame(dict(simulate.graph.degree).items(), columns=['key', 'degree'])
    mean_edges = np.array([simulate.graph.edges[e]['weight'] for e in simulate.graph.edges]).mean()
    row = {
        'edges': metrics['number-of-edges']['value'],
        'nodes': metrics['number-of-nodes']['value'],
        'wiring': metrics['wiring-cost']['value'],
        'wiring_norm': metrics['wiring-cost']['normalized_value'],
        'routing': metrics['routing-cost']['value'],
        'routing_norm': metrics['routing-cost']['normalized_value'],
        'fuel': metrics['fuel-cost']['value'],
        'fuel_norm': metrics['fuel-cost']['normalized_value'],
        'degree_mean': degree['degree'].mean(),
        'degree_std': degree['degree'].std(),
        'weight_mean': mean_edges,
        'distance_mean': math.pi / 2
    }
    logger.info(f'finished {i}, p={p}, N={N}')
    return row


def create_random_graphs():
    pool = ProcessPool(8)
    results = pool.map(lambda i: randomize(i), range(4000))

    df = pd.DataFrame(results)
    df.to_csv('optimize_results/random/random_full_more.csv', index=False)
    # ax = plt.subplot(projection='3d')
    # ax.scatter(wiring, routing, fuel)
    # ax.set_xlabel('wiring')
    # ax.set_ylabel('routing')
    # ax.set_zlabel('fuel')
    # plt.show()


def find_best_hyperparams_for_optimization():
    space = [
        skopt.space.Integer(10, 1000, name='n_gen'),
        skopt.space.Integer(10, 100, name='n_parents'),
        skopt.space.Real(0, 1, name='mutation_rate'),
    ]

    @skopt.utils.use_named_args(space)
    def objective(n_gen, n_parents, mutation_rate):
        go = GraphOptimizer(30, 60, 1, 0, 0, optimizer='genetic', n_gen=n_gen, n_parents=n_parents,
                            mutation_rate=mutation_rate)
        res = go.optimize()
        metrics = GraphFormatter(res).format_metrics()
        value = metrics['routing-cost']['value']
        print(f'current value: {value}')
        return value

    best = skopt.forest_minimize(objective, space, n_calls=30, n_random_starts=10, x0=[314, 80, 0.36646094592276784])

    print(f'best_result: {best.fun}')
    print(f'best_args: {best.x}')

    """
    best_result: 1.8860294117647058
    best_args: [314, 80, 0.36646094592276784]
    """


def optimize_multiple_times(map_func, times, nodes, edges, w, r, f, optimizer='genetic', overwrite=True):
    result_folder = f'optimize_results/{nodes}_{edges}/{w}_{r}_{f}'
    if overwrite:
        try:
            rmtree(result_folder)
        except:
            logger.warning(f'could not overwrite {result_folder}')
    try:
        os.makedirs(result_folder)
    except:
        logger.warning(f'could not create {result_folder}')

    def optimize(i):
        print(w, r, f)
        go = GraphOptimizer(nodes, edges, w, r, f, optimizer=optimizer)
        res = go.optimize()
        formatter = GraphFormatter(res)
        with open(f'optimize_results/{nodes}_{edges}/{w}_{r}_{f}/result_{i}.json', 'w') as F:
            F.write(json.dumps({'metrics': formatter.format_metrics(), 'chart': formatter.format_chart()}, indent=True))
        return res

    return map_func(optimize, range(times))


def create_optimized_graphs():

    pool = ProcessPool(8)
    times, nodes, edges = 100, 30, 60
    values = optimize_multiple_times(pool.map,  times, nodes, edges, 1, 0, 0)
    values += optimize_multiple_times(pool.map, times, nodes, edges, 0, 1, 0)
    values += optimize_multiple_times(pool.map, times, nodes, edges, 0, 0, 1)


def run_dataset(option):
    try:
        drc_local = DatasetsResultCache()
        drc_local.get_results(option)
    except Exception as e:
        print(f'ERROR {option}: {e}')


def delete_all_cached_datasets():
    json_files = glob.glob('./result_cache/*/*.json')
    print(json_files)
    for json_file in json_files:
        print(json_file)
        os.remove(json_file)


def run_all_datasets():
    # delete_all_cached_datasets()
    drc = DatasetsResultCache()
    pool = ProcessPool(8)
    for category in drc.options():
        print(category['name'])
        m = map(run_dataset, ['/'.join([category['name'], option['name']]) for option in category['options']])
        list(m)


if __name__ == '__main__':
    create_random_graphs()
    # run_all_datasets()
    # create_optimized_graphs()

