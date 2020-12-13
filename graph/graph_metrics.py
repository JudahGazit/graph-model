import itertools
import logging
import random

from bresenham import bresenham
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.csgraph
from bentley_ottmann.planar import segments_intersections
from scipy.spatial.distance import cityblock

from graph.metric_result import MetricResult, MetricBoundaries

logger = logging.getLogger('metrics')


def manhatten(u, v, n):
    u = u % n, int(u / n)
    v = v % n, int(v / n)
    return cityblock(u, v)


class CostBoundaries:
    def __init__(self, wiring=None, routing=None, fuel=None):
        self.wiring = wiring or MetricBoundaries()
        self.routing = routing or MetricBoundaries()
        self.fuel = fuel or MetricBoundaries()

class GraphMetrics:
    def __init__(self, graph_dataset=None, matrix=None, cost_boundaries=None):
        self.graph = graph_dataset.graph if graph_dataset else None  # nx.Graph
        self.distances = graph_dataset.distances if graph_dataset else None
        self.positions = graph_dataset.positions if graph_dataset else None
        self.all_shortest_paths = {True: None, False: None}

        if self.graph:
            self.number_of_nodes = self.graph.number_of_nodes()
            self.number_of_edges = self.graph.number_of_edges()
        else:
            self.number_of_nodes = matrix.shape[0]
            self.number_of_edges = int(np.count_nonzero(matrix) / 2)

        if matrix is not None:
            self.sparse_matrix = matrix
            self.igraph = igraph.Graph.Weighted_Adjacency(matrix.tolist(), mode=igraph.ADJ_UNDIRECTED) if self.number_of_nodes <= 500 else None
        else:
            self.sparse_matrix = nx.to_scipy_sparse_matrix(graph_dataset.graph)
            self.igraph = igraph.Graph.from_networkx(self.graph) if self.number_of_nodes <= 500 else None

        self.cost_boundaries = self.__create_cost_boundaries(cost_boundaries)


    def __group_by_matrix(self, mat: np.ndarray):
        unique_elements, counts = np.unique(mat, return_counts=True)
        mat = np.mat([unique_elements, counts]).transpose()
        gb = pd.DataFrame(mat, columns=['dist', 'count'])
        gb = gb[gb['dist'] > 0]
        return gb

    def __create_cost_boundaries(self, cost_boundaries):
        cost_boundaries = cost_boundaries if cost_boundaries is not None else CostBoundaries()
        if cost_boundaries.wiring.optimal_value is None:
            cost_boundaries.wiring.optimal_value = self.__wiring_cost_bounds(optimal=True)
        if cost_boundaries.wiring.worst_value is None:
            cost_boundaries.wiring.worst_value = self.__wiring_cost_bounds(optimal=False)
        if cost_boundaries.routing.optimal_value is None:
            cost_boundaries.routing.optimal_value = self.__optimal_routing_cost()
        if cost_boundaries.fuel.optimal_value is None:
            cost_boundaries.fuel.optimal_value = self.__optimal_fuel_cost()
        return cost_boundaries

    def __wiring_cost_bounds(self, optimal=True):
        total_number_of_possible_edges = self.number_of_nodes * (self.number_of_nodes - 1) / 2
        if self.number_of_nodes > 200:
            percentile = self.number_of_edges / total_number_of_possible_edges
            random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            distances = sorted([self.distances(u, v) for u, v in itertools.combinations(random_nodes, 2)], reverse=not optimal)
            number_of_random_edges = len(distances)
            sum_of_percentile_samples = sum(distances[:max([int(number_of_random_edges * percentile), 1])])
            return sum_of_percentile_samples * (total_number_of_possible_edges / number_of_random_edges)
        else:
            distances = sorted([self.distances(u, v) for u, v in itertools.combinations(range(self.number_of_nodes), 2)], reverse=not optimal)
            return sum(distances[:min([self.number_of_edges, len(distances)])])

    def __optimal_routing_cost(self):
        number_of_pairs = self.number_of_nodes * (self.number_of_nodes - 1) / 2
        number_of_pairs_in_distance_1 = self.number_of_edges
        number_of_pairs_in_distance_2 = number_of_pairs - self.number_of_edges
        mean_degree = (number_of_pairs_in_distance_1 + 2 * number_of_pairs_in_distance_2) / number_of_pairs
        return mean_degree

    def __optimal_fuel_cost(self):
        if self.number_of_nodes > 200:
            random_nodes = random.sample(range(self.number_of_nodes), min([200, self.number_of_nodes]))
            mean_distance = np.array([self.distances(u, v) for u, v in itertools.combinations(random_nodes, 2)]).mean()
            return mean_distance
        else:
            distance_mat = [[self.distances(u, v) if u != v else 0 for u in range(self.number_of_nodes)] for v in range(self.number_of_nodes)]
            distance_graph = igraph.Graph.Weighted_Adjacency(distance_mat, mode=igraph.ADJ_UNDIRECTED)
            shortest_paths = np.mat(distance_graph.shortest_paths(None, weights='weight'))
            return shortest_paths[np.nonzero(shortest_paths)].mean()


    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        if self.all_shortest_paths[weight] is None:
            logger.debug(f'shortest path started - weight={weight}')
            indices = random.sample(range(self.number_of_nodes), 1000) if self.number_of_nodes > 1000 else None
            if self.igraph is not None:
                all_pairs_shortest_path = self.igraph.shortest_paths(indices, weights='weight' if weight else None)
            else:
                all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                             unweighted=not weight, indices=indices)
            logger.debug('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            logger.debug('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def wiring_cost(self):
        logger.debug('start wiring cost')
        result = MetricResult(self.sparse_matrix.sum() / 2, self.cost_boundaries.wiring)
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        df = self.all_path_lengths(False)
        routing_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = MetricResult(routing_cost, self.cost_boundaries.routing)
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        df = self.all_path_lengths(True)
        fuel_cost = (df['dist'] * df['count']).sum() / (df['count']).sum()
        result = MetricResult(fuel_cost, self.cost_boundaries.fuel)
        logger.debug('end fuel cost')
        return result
    #
    # def collision_cost(self):
    #     edges = self.igraph.get_edgelist()
    #     collision_count = 0
    #     for e1, e2 in itertools.combinations(edges, 2):
    #         if len(set(e1 + e2)) == 4:
    #             [x0, y0], [x1, y1] = self.positions(e1[0]), self.positions(e1[1])
    #             [x2, y2], [x3, y3] = self.positions(e2[0]), self.positions(e2[1])
    #             m1 = (y1 - y0) / (x1 - x0) if x0 != x1 else None
    #             m2 = (y3 - y2) / (x3 - x2) if x2 != x3 else None
    #             if m1 is None or m2 is None:
    #                 y = None
    #                 if m2 is not None:
    #                     x = x0
    #                     y = (x - x2) * m2 + y2
    #                 elif m1 is not None:
    #                     x = x2
    #                     y = (x - x0) * m1 + y0
    #                 if y is not None and min([y0, y1]) < y < max([y0, y1]) and min([y2, y3]) < y < max([y2, y3]):
    #                     collision_count += 1
    #             elif m1 != m2:
    #                 x = (x0 * m1 - x2 * m2 + y2 - y0) / (m1 - m2)
    #                 if min([x0, x1]) < x < max([x0, x1]) and min([x2, x3]) < x < max([x2, x3]):
    #                     collision_count += 1
    #     return MetricResult(collision_count)

    def collision_cost(self, grid_scale=500):
        if self.positions is not None:
            n = int(np.sqrt(self.number_of_nodes))
            mat = np.asmatrix(np.zeros([grid_scale * (n - 1) + 1, grid_scale * (n - 1) + 1]))
            edges = self.igraph.get_edgelist()
            for u, v in edges:
                pos_u = np.array(self.positions(u)) * grid_scale
                pos_v = np.array(self.positions(v)) * grid_scale
                gridded_line = tuple(bresenham(*pos_u, *pos_v))
                # print(pos_u, pos_v, gridded_line)
                for x, y in gridded_line:
                    mat[y, x] += 1
            # print(mat[::-1])
            mat[::grid_scale, ::grid_scale] = 0
            # print(mat[::-1])
            intersections = np.sum(mat[mat > 1] - 1)
            return MetricResult(intersections)
        return MetricResult(-1)