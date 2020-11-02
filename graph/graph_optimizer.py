import json
from glob import glob
import numpy as np
from pathos.multiprocessing import ProcessPool

from graph.graph_formatter import GraphFormatter
from graph.graph_optimizers.genetic_algorithm import GeneticAlgorithm
from graph.graph_optimizers.random_optimum import RandomOptimum
from graph.graph_optimizers.simulated_annealing import SimulatedAnnealing
import skopt
import matplotlib.pyplot as plt

POOL_SIZE = 4

OPTIMIZERS = {
    'annealing': SimulatedAnnealing,
    'random': RandomOptimum,
    'genetic': GeneticAlgorithm,
}


class GraphOptimizer:
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method='minimize',
                 optimizer='genetic', **kwargs):
        self.optimizer = OPTIMIZERS[optimizer](num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method,
                                               **kwargs)

    def optimize(self):
        return self.optimizer.optimize()




if __name__ == '__main__':
    def optimize(i):
        go = GraphOptimizer(100, 500, 0, 0, 1, optimizer='genetic')
        res = go.optimize()
        formatter = GraphFormatter(res)
        with open(f'optimize_results/metrics_genetic2_{i}_fuel.json', 'w') as F:
            F.write(json.dumps({'metrics': formatter.format_metrics(), 'chart': formatter.format_chart()}, indent=True))
        return res

    pool = ProcessPool(8)

    space = [
        skopt.space.Integer(10, 1000, name='n_gen'),
        skopt.space.Integer(10, 100, name='n_parents'),
        skopt.space.Real(0, 1, name='mutation_rate'),
    ]

    @skopt.utils.use_named_args(space)
    def objective(n_gen, n_parents, mutation_rate):
        go = GraphOptimizer(30, 60, 1, 0, 0, optimizer='genetic', n_gen=n_gen, n_parents=n_parents, mutation_rate=mutation_rate)
        res = go.optimize()
        metrics = GraphFormatter(res).format_metrics()
        value = metrics['routing-cost']['value']
        print(f'current value: {value}')
        return value



    # best = skopt.forest_minimize(objective, space, n_calls=30, n_random_starts=10, x0=[314, 80, 0.36646094592276784])

    # print(f'best_result: {best.fun}')
    # print(f'best_args: {best.x}')

    """
    best_result: 1.8860294117647058
    best_args: [314, 80, 0.36646094592276784]
    """

    # values = pool.map(optimize, range(100))
    files = glob('optimize_results/*_genetic2_*_wiring*.json')
    plt.title('wiring')
    data = [json.loads(open(filename).read()) for filename in files]
    ys = []
    for dataset in data:
        y = dataset['chart']['edge-length-dist']['y']
        y = [y[i] + y[i+1] for i in range(0, len(y) -1, 2)]
        s = sum(y)
        y = np.array(y) / s
        ys.append(y)
        plt.plot(range(len(y)), y)
    ymean = np.mean(np.mat(list(zip(*ys))), axis=1)[:, 0]
    plt.plot(range(len(ymean)), ymean, color='black', linewidth=5)
    plt.show()