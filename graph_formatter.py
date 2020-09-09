import pandas as pd


class GraphFormatter:
    def __init__(self, graph):
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

    def format_graph(self):
        return self.df.to_dict('records')

    def format_chart(self):
        df = self.df.copy()
        df['bins'] = pd.cut(df['weight'], 20).apply(str)
        df = df.groupby('bins', as_index=False).agg({'weight': 'count'})
        df.columns = ['x', 'y']
        result = df.to_dict('list')
        return result