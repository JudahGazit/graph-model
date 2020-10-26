import json

from pathos.multiprocessing import ProcessPool

from graph.graph_formatter import GraphFormatter
from graph.graph_optimizers.genetic_algorithm import GeneticAlgorithm
from graph.graph_optimizers.random_optimum import RandomOptimum
from graph.graph_optimizers.simulated_annealing import SimulatedAnnealing

POOL_SIZE = 4

OPTIMIZERS = {
    'annealing': SimulatedAnnealing,
    'random': RandomOptimum,
    'genetic': GeneticAlgorithm,
}


class GraphOptimizer:
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method='minimize',
                 optimizer='genetic', *args, **kwargs):
        self.optimizer = OPTIMIZERS[optimizer](num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method,
                                               *args, **kwargs)

    def optimize(self):
        return self.optimizer.optimize()


if __name__ == '__main__':
    pool = ProcessPool(8)


    def optimize(i):
        go = GraphOptimizer(30, 60, 0, 1, 0, optimizer='genetic')
        res = go.optimize()
        metrics = GraphFormatter(res).format_metrics()
        with open(f'optimize_results/metrics_genetic_{i}.json', 'w') as F:
            F.write(json.dumps(metrics, indent=True))
        return res


    values = pool.map(optimize, range(8))
