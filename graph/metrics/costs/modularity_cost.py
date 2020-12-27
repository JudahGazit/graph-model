import numpy as np

from graph.metrics.Metric import MetricBoundaries, Metric
from graph.metrics.costs.icost import ICost


class ModularityCost(ICost):
    def categorise_node(self, node, number_of_modules):
        n = int(np.sqrt(self.number_of_nodes))
        module_size = n // number_of_modules
        position = np.asarray(self.positions(node)) // module_size
        return int(''.join(position.astype('str').tolist()))

    def cost(self):
        number_of_modules = 2
        categorise_nodes = np.array([self.categorise_node(node, number_of_modules) for node in range(self.number_of_nodes)])
        edge_sources, edge_targets = np.nonzero(self.adjacency)
        crossings_count = np.count_nonzero(categorise_nodes[edge_sources] != categorise_nodes[edge_targets])
        return Metric(crossings_count / 2, self.boundaries)


    @property
    def boundaries(self) -> MetricBoundaries:
        return MetricBoundaries(10)