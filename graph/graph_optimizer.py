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
    def __init__(self, num_nodes, num_edges, factors, method='maximize', cost_type='circular',
                 optimizer='genetic', **kwargs):
        self.optimizer = OPTIMIZERS[optimizer](num_nodes, num_edges, factors, method, cost_type, **kwargs)

    def optimize(self):
        return self.optimizer.optimize()

    @property
    def graph_cost(self):
        return self.optimizer.graph_cost
