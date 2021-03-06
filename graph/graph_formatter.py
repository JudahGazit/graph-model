import logging
import random

import networkx as nx
import pandas as pd

from graph.graph_metrics import GraphMetrics
from graph.metric_result import MetricResult
from graph.graph_categories.graph_categories import GraphDataset


logger = logging.getLogger('formatter')

class GraphFormatter:
    def __init__(self, graph: GraphDataset, topology='circular'):
        self.graph = graph.graph
        self.distances = graph.distances
        self.df = self.format_graph_to_df(self.graph)
        self.distances_bins = self.__distances_bins(self.graph, self.distances)
        self.graph_metrics = GraphMetrics(graph, topology=topology)

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
        if bins and bins_type == 'data':
            df['bins'] = pd.cut(df[group_column], 20, include_lowest=True, right=False).apply(str)
            df = df.groupby('bins', as_index=False).agg({agg_column: method})
        elif bins and bins_type == 'distances':
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
        degree_histogram = nx.degree_histogram(self.graph)
        result = {'x': list(range(len(degree_histogram)))[:500],
                  'y': degree_histogram[:500]}
        return result

    def __node_path_length_dist_chart(self, weight=False, bins=False):
        df = self.graph_metrics.all_path_lengths(weight)
        df = self.__agg_df(df, "dist", "count", bins, method="sum")
        result = df.to_dict('list')
        return result

    def format_graph(self):
        return self.df.to_dict('records')

    def format_graph_sample(self, max_depth=10):
        logger.debug('start formatting graph sample')
        random_node = random.choice(list(self.graph.nodes))
        nodes_in_reach = nx.algorithms.bfs_tree(self.graph, random_node, depth_limit=max_depth).nodes()
        sampled_graph = self.graph.subgraph(list(nodes_in_reach)[:500])
        logger.debug('end formatting graph sample')
        return self.format_graph_to_df(sampled_graph).to_dict('records')

    def __heaviest_edge_weight(self, graph):
        weights = [graph.edges[edge]['weight'] for edge in graph.edges]
        return max(weights)

    def format_chart(self):
        logger.debug('start formatting charts')
        charts = {
            'edge-length-dist': self.__edge_length_dist_chart(),
            'edge-length-dist-dbins': self.__edge_length_dist_chart('distances'),
            'degree-histogram': self.__degree_dist_chart(),
            'node-path-len-dist': self.__node_path_length_dist_chart(),
            'nodes-distance-dist': self.__node_path_length_dist_chart(True, True)
        }
        logger.debug('end formatting charts')
        return charts

    def format_metrics(self):
        logger.debug('start formatting metrics')
        number_of_nodes = self.graph.number_of_nodes()
        number_of_edges = self.graph.number_of_edges()
        metrics = {
            'number-of-nodes': MetricResult(number_of_nodes, number_of_nodes),
            'number-of-edges': MetricResult(number_of_edges, number_of_edges),
            'wiring-cost': self.graph_metrics.wiring_cost(),
            'routing-cost': self.graph_metrics.routing_cost(),
            'fuel-cost': self.graph_metrics.fuel_cost(),
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
