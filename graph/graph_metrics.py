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

    def collision_cost(self, grid_scale=2):
        if self.positions is not None:
            node_positions = [grid_scale * np.array(self.positions(i)) for i in range(self.number_of_nodes)]
            edges = pd.DataFrame([[row] for row in self.igraph.get_edgelist()], columns=['edge'])
            edges['path'] = edges['edge'].apply(lambda row: tuple(bresenham(*node_positions[row[0]], *node_positions[row[1]])),)
            edges = edges.explode('path').groupby('path', as_index=False).agg(set)
            edges = edges[edges['path'].apply(lambda point: point[0] % grid_scale != 0 or point[1] % grid_scale != 0)]
            edges = edges[edges['edge'].str.len() > 1]
            edges['edge'] = edges['edge'].apply(lambda intersection: [(*u, *v) for u, v in itertools.combinations(intersection, 2) if len({*u, *v}) == 4])
            edges = edges.explode('edge').dropna()
            intersections = edges['edge'].drop_duplicates().size
            return MetricResult(intersections, MetricBoundaries(100))
        return MetricResult(-1)