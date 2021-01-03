import logging
import random

import networkx as nx
import pandas as pd

from graph.graph_dataset import GraphDataset
from graph.metrics.Metric import Metric
from graph.metrics.costs.fuel_cost import FuelCost
from graph.metrics.costs.icost import ICost
from graph.metrics.costs.intersection_cost import IntersectionCost
from graph.metrics.costs.modularity_cost import ModularityCost
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.metrics.costs.routing_cost import RoutingCost
from graph.metrics.costs.volume_cost import VolumeCost
from graph.metrics.costs.wiring_cost import WiringCost

logger = logging.getLogger('formatter')


class GraphFormatter:
    def __init__(self, graph_dataset: GraphDataset):
        self.graph_dataset = graph_dataset
        self.df = self.format_graph_to_df(self.graph_dataset.nx_graph)
        self.distances_bins = self.__distances_bins(self.graph_dataset.nx_graph, self.graph_dataset.distances.item)

    def format_graph_to_df(self, graph):
        result = []
        for i, js in graph.adj.items():
            for j, w in js.items():
                if i < j:
                    result.append({'source': i, 'target': j, 'weight': w['weight']})
        df = pd.DataFrame(result)
        return df

    def __agg_df(self, df, group_column, agg_column, bins, bins_type='data', method='count'):
        df = df.copy()
        df = df[df[group_column] < float('inf')]
        if bins and bins_type == 'data':
            df['bins'] = pd.cut(df[group_column], 20, include_lowest=True, right=False).apply(str)
            df = df.groupby('bins', as_index=False).agg({agg_column: method})
        elif bins and bins_type == 'distances_f':
            df['bins'] = pd.cut(df[group_column], self.distances_bins[1], include_lowest=True, right=False).apply(str)
            df = df.groupby('bins', as_index=False).agg({agg_column: method})
            df = df.merge(self.distances_bins[0], on=['bins'])
            df[agg_column] = (df[agg_column] / df['dist']).fillna(0)
            df = df[['bins', agg_column]]
        else:
            df = df.groupby(group_column, as_index=False).agg({agg_column: method})
        df.columns = ['x', 'y']
        return df

    def __edge_length_dist_chart(self, bins_type='data'):
        df = self.__agg_df(self.df, "weight", "weight", bins=True, bins_type=bins_type)
        result = df.to_dict('list')
        return result

    def __degree_dist_chart(self):
        degree_histogram = nx.degree_histogram(self.graph_dataset.nx_graph)
        result = {'x': list(range(len(degree_histogram)))[:500],
                  'y': degree_histogram[:500]}
        return result

    def __node_path_length_dist_chart(self, weight=False, bins=False):
        df = ICost(self.graph_dataset).all_path_lengths(weight)
        df = self.__agg_df(df, "dist", "count", bins, method="sum")
        result = df.to_dict('list')
        return result

    def format_graph(self):
        return self.df.to_dict('records')

    def format_graph_sample(self, max_depth=10):
        logger.debug('start formatting graph_dataset sample')
        random_node = random.choice(range(self.graph_dataset.number_of_nodes))
        nodes_in_reach = nx.algorithms.bfs_tree(self.graph_dataset.nx_graph, random_node, depth_limit=max_depth).nodes()
        sampled_graph = self.graph_dataset.nx_graph.subgraph(list(nodes_in_reach)[:500])
        logger.debug('end formatting graph_dataset sample')
        return self.format_graph_to_df(sampled_graph).to_dict('records')

    def __heaviest_edge_weight(self, graph):
        weights = [graph.edges[edge]['weight'] for edge in graph.edges]
        return max(weights)

    def format_chart(self):
        logger.debug('start formatting charts')
        charts = {
            'edge-length-dist': self.__edge_length_dist_chart(),
            'edge-length-dist-dbins': self.__edge_length_dist_chart('distances_f'),
            'degree-histogram': self.__degree_dist_chart(),
            'node-path-len-dist': self.__node_path_length_dist_chart(),
            'nodes-distance-dist': self.__node_path_length_dist_chart(True, True),
            'degree-edge-distance-correlation': self.__node_degree_and_edge_length_correlation(),
            'degree-and-degree-of-neighbours': self.__degree_and_degree_of_neighbours(),
            'triangles-hist': self.__triangles_hist()
        }
        logger.debug('end formatting charts')
        return charts

    def format_metrics(self):
        metrics = {
            'number-of-nodes': Metric(self.graph_dataset.number_of_nodes),
            'number-of-edges': Metric(self.graph_dataset.number_of_edges),
            'wiring-cost': WiringCost(self.graph_dataset).cost(),
            'routing-cost': RoutingCost(self.graph_dataset).cost(),
            'fuel-cost': FuelCost(self.graph_dataset).cost(),
            'collision-cost': IntersectionCost(self.graph_dataset).cost(),
            'modularity-cost': ModularityCost(self.graph_dataset).cost(),
            'volume-cost': VolumeCost(self.graph_dataset).cost(),
            'resistance-cost': ResistanceCost(self.graph_dataset).cost(),
        }
        metrics = {name: metric.to_dict() for name, metric in metrics.items()}
        logger.debug('end formatting metrics')
        return metrics

    def __distances_bins(self, graph, distances, bins=20):
        num_nodes = len(graph.nodes)
        sample_size = min([200, num_nodes])
        nodes = pd.DataFrame(random.sample(range(num_nodes), sample_size), columns=['id'])
        nodes['key'] = 0
        pairs = nodes.merge(nodes, on=['key']).query('id_x != id_y')
        pairs['dist'] = pairs[['id_x', 'id_y']].apply(lambda p: distances(*p), axis=1)
        factor, bins = pd.cut(pairs['dist'], bins, retbins=True, include_lowest=True, right=False)
        pairs['bins'] = factor.apply(str)
        pairs = pairs.groupby(['bins'], as_index=False).agg({'dist': 'count'})
        pairs['dist'] = (num_nodes * (num_nodes - 1) / 2) * pairs['dist'] / pairs['dist'].sum()
        return pairs, bins

    def __node_degree_and_edge_length_correlation(self):
        degrees = [degree for node, degree in sorted(self.graph_dataset.nx_graph.degree)]
        node_average_edge_dist = nx.to_numpy_array(self.graph_dataset.nx_graph).mean(axis=0)
        return {
            'x': degrees,
            'y': node_average_edge_dist.tolist(),
            'type': 'circle'
        }

    def __degree_and_degree_of_neighbours(self):
        degrees = pd.DataFrame(self.graph_dataset.nx_graph.degree, columns=['node', 'degree'])
        edges = pd.DataFrame(self.graph_dataset.nx_graph.edges, columns=['src', 'dst'])
        edges = edges.merge(degrees, left_on=['src'], right_on=['node']).merge(degrees, left_on=['dst'],
                                                                               right_on=['node'])
        edges = edges[['src', 'dst', 'degree_x', 'degree_y']]
        edges = edges.groupby('src', as_index=False).mean().groupby('degree_x', as_index=False).mean()
        return {
            'x': edges['degree_x'].tolist(),
            'y': edges['degree_y'].tolist()
        }

    def __triangles_hist(self):
        triangles = pd.DataFrame(nx.triangles(self.graph_dataset.nx_graph).values(), columns=['x'])
        triangles['y'] = 1
        return self.__agg_df(triangles, 'x', 'y', True).to_dict('list')
