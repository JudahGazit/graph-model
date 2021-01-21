import itertools

import numpy as np

from graph.graph_dataset import GraphDataset
from graph.metrics.Metric import MetricBoundaries, Metric
from graph.metrics.costs.icost import INeighboursCost


class ResistanceCost(INeighboursCost):
    def __init__(self, graph_dataset: GraphDataset, boundaries=None):
        super().__init__(graph_dataset, boundaries)
        self.node_resistance = 1
        self.__omega = None

    def cost(self):
        omega = self.omega()
        if omega is not None:
            resistance = omega.sum() / 2
            metric = Metric(resistance, self._boundaries)
            return metric
        return Metric(float('inf'), self.boundaries)

    def omega(self):
        if self.__omega is None:
            num_nodes = self.graph_dataset.number_of_nodes
            if self.graph_dataset.is_connected:
                laplacian = self.__create_laplacian()
                gamma = self.__calculate_gamma(laplacian, num_nodes)
                omega = self.__calculate_omega_from_gamma(gamma)
                self.__omega = omega
        return self.__omega

    def __calculate_omega_from_gamma(self, gamma):
        omega = -2 * gamma
        omega += np.diag(gamma)
        omega += np.asmatrix(np.diag(gamma)).transpose()
        return omega

    def __calculate_gamma(self, laplacian, num_nodes):
        gamma = laplacian + 1 / num_nodes
        gamma = np.linalg.pinv(gamma, hermitian=True)
        return gamma

    def _edge_conductance(self, adjacency):
        if self.graph_dataset.widths is not None:
            adjacency = np.multiply(self.graph_dataset.widths, adjacency)
        else:
            adjacency = adjacency
        edge_conductance = np.divide(adjacency,
                                     np.square(self.graph_dataset.distances) + 2 * self.node_resistance * adjacency,
                                     out=np.zeros_like(adjacency),
                                     where=self.graph_dataset.distances != 0)
        return edge_conductance

    def __create_laplacian(self):
        edge_conductance = self._edge_conductance(self.graph_dataset.adjacency)
        laplacian = -edge_conductance
        np.fill_diagonal(laplacian, edge_conductance.sum(axis=1))
        return laplacian

    def __resistance_if_add_or_remove(self, plus_or_minus=1):
        np.set_printoptions(2)
        adjacency = np.maximum(self.graph_dataset.adjacency + plus_or_minus * self.graph_dataset.distances,
                               np.zeros_like(self.graph_dataset.adjacency))
        omega = self.omega()
        n = self.graph_dataset.number_of_nodes
        delta = self._edge_conductance(adjacency) - self._edge_conductance(self.graph_dataset.adjacency)
        kfg = omega.sum() / 2

        omega_row_sum = omega.sum(1)
        omega_sum_diff = np.square(np.array(np.meshgrid(-omega_row_sum, omega_row_sum)).sum(0))
        omega_squared_row_sum = np.square(omega).sum(1)
        omega_squared_diff = np.array(np.meshgrid(omega_squared_row_sum, omega_squared_row_sum)).sum(0) - 2 * np.matmul(omega, omega.transpose())

        factor = (n * np.multiply(delta, omega_squared_diff) - np.multiply(delta, omega_sum_diff)) / (4 * (1 + np.multiply(delta, omega)))
        kfg_options = kfg - factor
        return kfg_options

    def costs_if_add(self):
        return self.__resistance_if_add_or_remove(1)

    def costs_if_remove(self):
        return self.__resistance_if_add_or_remove(-1)

    @property
    def boundaries(self) -> MetricBoundaries:
        return MetricBoundaries()
