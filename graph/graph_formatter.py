import random
from collections import namedtuple
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse

from graph.MetricResult import MetricResult
from graph.graph_categories.graph_categories import GraphDataset


class GraphFormatter:
    def __init__(self, graph: GraphDataset):
        self.graph = graph.graph
        self.distances = graph.distances
        self.df = self.format_graph_to_df(self.graph)
        self.sparse_matrix = nx.to_scipy_sparse_matrix(self.graph, dtype=np.int)
        self.all_shortest_paths = {True: None, False: None}
        self.normalization_factor = self.__heaviest_edge_weight(self.graph)
        self.distances_bins = self.__distances_bins(self.graph, self.distances)

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
        df = self.__all_path_lengths(weight)
        df = self.__agg_df(df, "dist", "count", bins, method="sum")
        result = df.to_dict('list')
        return result

    def __group_by_matrix(self, df):
        gb = pd.DataFrame(df.flatten())
        gb.columns = ['dist']
        gb['count'] = 0
        gb = gb[gb['dist'] < float('inf')]
        gb = gb.groupby(['dist'], as_index=False).agg({'count': 'count'})
        return gb

    def __all_path_lengths(self, weight=False):
        if self.all_shortest_paths[weight] is None:
            print(f'shortest path started - weight={weight}')
            indices = random.sample(list(self.graph.nodes()), min([1000, self.graph.number_of_nodes()]))
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                         unweighted=not weight, indices=indices)
            print('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            print('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def __wiring_cost_metric(self):
        return MetricResult(self.graph.size("weight"), self.normalization_factor)

    def __routing_cost_metric(self):
        df = self.__all_path_lengths()
        return MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum())

    def __fuel_cost_metric(self):
        df = self.__all_path_lengths(True)
        return MetricResult((df['dist'] * df['count']).sum() / (df['count']).sum(),
                            self.normalization_factor)

    def format_graph(self):
        return self.df.to_dict('records')

    def format_graph_sample(self, max_depth=10):
        random_node = random.choice(list(self.graph.nodes))
        nodes_in_reach = nx.algorithms.bfs_tree(self.graph, random_node, depth_limit=max_depth).nodes()
        sampled_graph = self.graph.subgraph(list(nodes_in_reach)[:500])
        return self.format_graph_to_df(sampled_graph).to_dict('records')

    def __heaviest_edge_weight(self, graph):
        weights = [graph.edges[edge]['weight'] for edge in graph.edges]
        return max(weights)

    def format_chart(self):
        charts = {
            'edge-length-dist': self.__edge_length_dist_chart(),
            'edge-length-dist-dbins': self.__edge_length_dist_chart('distances'),
            'degree-histogram': self.__degree_dist_chart(),
            'node-path-len-dist': self.__node_path_length_dist_chart(),
            'nodes-distance-dist': self.__node_path_length_dist_chart(True, True)
        }
        return charts

    def format_metrics(self):
        metrics = {
            'number-of-nodes': MetricResult(self.graph.number_of_nodes()),
            'number-of-edges': MetricResult(self.graph.number_of_edges()),
            'wiring-cost': self.__wiring_cost_metric(),
            'routing-cost': self.__routing_cost_metric(),
            'fuel-cost': self.__fuel_cost_metric(),
        }
        metrics = {name: metric.to_dict() for name, metric in metrics.items()}
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