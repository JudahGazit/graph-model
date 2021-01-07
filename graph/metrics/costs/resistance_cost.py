import networkx as nx
import itertools

import numpy as np
from graph.metrics.Metric import MetricBoundaries, Metric
from graph.metrics.costs.icost import ICost


class ResistanceCost(ICost):
    def cost(self):
        num_nodes = self.graph_dataset.number_of_nodes
        if self.graph_dataset.is_connected:
            laplacian = self.__create_laplacian()
            gamma = self.__calculate_gamma(laplacian, num_nodes)
            omega = self.__calculate_omega(gamma)
            resistance = omega[np.triu_indices_from(omega, 1)].mean()
            return Metric(resistance, self.boundaries)
        return Metric(float('inf'), self.boundaries)

    def __calculate_omega(self, gamma):
        omega = -2 * gamma
        omega += np.diag(gamma)
        omega += np.asmatrix(np.diag(gamma)).transpose()
        return omega

    def __calculate_gamma(self, laplacian, num_nodes):
        gamma = laplacian + 1 / num_nodes
        gamma = np.linalg.pinv(gamma, hermitian=True)
        return gamma

    def __create_laplacian(self):
        if self.graph_dataset.widths is not None:
            adjacency = np.multiply(self.graph_dataset.widths, self.graph_dataset.adjacency)
        else:
            adjacency = self.graph_dataset.adjacency
        edge_conductance = np.divide(adjacency, np.square(self.graph_dataset.distances),
                                     out=np.zeros_like(self.graph_dataset.adjacency),
                                     where=self.graph_dataset.distances != 0)
        laplacian = -edge_conductance
        np.fill_diagonal(laplacian, edge_conductance.sum(axis=1))
        return laplacian

    @property
    def boundaries(self) -> MetricBoundaries:
        return MetricBoundaries(0.18)
