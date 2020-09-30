import networkx as nx
import pandas as pd


class GraphFormatter:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.df = self.format_graph_to_df()

    def format_graph_to_df(self):
        result = []
        for i, js in self.graph.adj.items():
            for j, w in js.items():
                if i < j:
                    result.append({'source': i, 'target': j, 'weight': w['weight']})
        df = pd.DataFrame(result)
        return df

    def __agg_df(self, df, group_column, agg_column, bins):
        df = df.copy()
        if bins:
            df['bins'] = pd.cut(df[group_column], 20).apply(str)
            df = df.groupby('bins', as_index=False).agg({agg_column: 'count'})
        else:
            df = df.groupby(group_column, as_index=False).agg({agg_column: 'count'})
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
        df = self.__agg_df(df, "dist", "from", bins)
        result = df.to_dict('list')
        return result

    def __all_path_lengths(self, weight=False):
        weight = "weight" if weight else None
        all_pairs_shortest_path = list(nx.shortest_path_length(self.graph, weight=weight))
        df = [{'from': x, 'to': y, 'dist': d} for x, ys in all_pairs_shortest_path for y, d in ys.items()]
        df = pd.DataFrame(df)
        df = df[df['from'] < df['to']]
        return df

    def __wiring_cost_metric(self):
        return self.graph.size("weight")

    def __routing_cost_metric(self):
        df = self.__all_path_lengths()
        return df['dist'].mean()

    def __fuel_cost_metric(self):
        df = self.__all_path_lengths(True)
        return df['dist'].mean()

    def format_graph(self):
        return self.df.to_dict('records')

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