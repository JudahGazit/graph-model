import functools
import random

import networkx as nx
import pandas as pd
import scipy.sparse
from tqdm import tqdm


class GraphFormatter:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.df = self.format_graph_to_df(graph)
        self.sparse_matrix = nx.to_scipy_sparse_matrix(self.graph)
        self.all_shortest_paths = {True: None, False: None}

    def format_graph_to_df(self, graph):
        result = []
        for i, js in graph.adj.items():
            for j, w in js.items():
                if i < j:
                    result.append({'source': i, 'target': j, 'weight': w['weight']})
        df = pd.DataFrame(result)
        return df

    def __agg_df(self, df, group_column, agg_column, bins, method='count'):
        df = df.copy()
        if bins:
            df['bins'] = pd.cut(df[group_column], 20).apply(str)
            df = df.groupby('bins', as_index=False).agg({agg_column: method})
        else:
            df = df.groupby(group_column, as_index=False).agg({agg_column: method})
        df.columns = ['x', 'y']
        return df

    def __edge_length_dist_chart(self):
        df = self.__agg_df(self.df, "weight", "weight", True)
        result = df.to_dict('list')
        return result

    def __degree_dist_chart(self):
        degree_histogram = nx.degree_histogram(self.graph)
        result = {'x': list(range(len(degree_histogram))),
                  'y': degree_histogram}
        return result

    def __node_path_length_dist_chart(self, weight=False, bins=False):
        df = self.__all_path_lengths(weight)
        df = self.__agg_df(df, "dist", "count", bins, method="sum")
        result = df.to_dict('list')
        return result

    def __group_by_matrix(self, df):
        gb = {}
        for row in tqdm(range(len(df))):
            for col in range(row, len(df)):
                cell = df[row][col]
                if cell < float('inf'):
                    cell = int(cell)
                    if cell not in gb:
                        gb[cell] = 0
                    gb[cell] += 1
        gb = pd.DataFrame(gb.items(), columns=['dist', 'count'])
        return gb

    def __all_path_lengths(self, weight=False):
        if self.all_shortest_paths[weight] is None:
            print(f'shortest path started - weight={weight}')
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.sparse_matrix, directed=False,
                                                                         unweighted=not weight)
            print('shortest path is done')
            gb = self.__group_by_matrix(all_pairs_shortest_path)
            print('group by is done')
            self.all_shortest_paths[weight] = gb
        return self.all_shortest_paths[weight]

    def __wiring_cost_metric(self):
        return self.graph.size("weight")

    def __routing_cost_metric(self):
        df = self.__all_path_lengths()
        return (df['dist'] * df['count']).mean()

    def __fuel_cost_metric(self):
        df = self.__all_path_lengths(True)
        return (df['dist'] * df['count']).mean()

    def format_graph(self):
        return self.df.to_dict('records')

    def format_graph_sample(self, max_nodes=500):
        sampled_nodes = random.sample(self.graph.nodes, max_nodes)
        sampled_graph = self.graph.subgraph(sampled_nodes)
        return self.format_graph_to_df(sampled_graph).to_dict('records')

    def format_chart(self):
        charts = {
            'edge-length-dist': self.__edge_length_dist_chart(),
            'degree-histogram': self.__degree_dist_chart(),
            'node-path-len-dist': self.__node_path_length_dist_chart(),
            'nodes-distance-dist': self.__node_path_length_dist_chart(True, True)
        }
        return charts

    def format_metrics(self):
        return {
            'wiring-cost': self.__wiring_cost_metric(),
            'routing-cost': self.__routing_cost_metric(),
            'fuel-cost': self.__fuel_cost_metric(),
        }