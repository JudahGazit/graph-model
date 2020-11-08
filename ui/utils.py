import math
from functools import reduce

import networkx as nx
import pandas as pd
import pydot
import streamlit as st
import altair as alt

CHART_NAME_MAPPING = {
    'edge-length-dist': 'Fiber Length Histogram',
    'edge-length-dist-dbins': 'Fiber Length Histogram (normalized by distances bin)',
    'degree-histogram': 'Node Degree Histogram',
    'node-path-len-dist': 'Unweighted Shortest Path Distance Histogram',
    'nodes-distance-dist': 'Weighted Shortest Path Distance Histogram'
}


def display_graph(graph: nx.Graph, node_method=None, edge_method=None, engine='neato'):
    for i, node in enumerate(sorted(graph.nodes)):
        if node_method:
            node_method(graph, node, i)
        graph.nodes[node]['width'] = "0.1"
        graph.nodes[node]['height'] = "0.1"
        graph.nodes[node]['label'] = ""

    for i, edge in enumerate(sorted(graph.edges)):
        if edge_method:
            edge_method(graph, edge, i)

    pdot = nx.nx_pydot.to_pydot(graph)
    pdot.set('layout', engine)
    st.subheader('Network')
    st.graphviz_chart(pdot.to_string(), False)


def display_graph_in_circle(graph: nx.Graph):
    def node_method(graph, node, node_index):
        pos_x = round(math.cos(2 * node_index * math.pi / len(graph.nodes)) * 3, 3)
        pos_y = round(math.sin(2 * node_index * math.pi / len(graph.nodes)) * 3, 3)
        graph.nodes[node]['pos'] = f'{pos_x},{pos_y}!'

    display_graph(graph, node_method)


def display_graph_naturally(graph: nx.Graph):
    max_edge_weight = max(graph.edges(data=True), key=lambda x: x[2]['weight'])[2]['weight']

    def edge_method(graph, edge, node_index):
        graph.edges[edge]['len'] = 30 * graph.edges[edge]['weight'] / max_edge_weight

    display_graph(graph, edge_method=edge_method, engine='fdp')


def display_metrics(metrics):
    metric_df = [(metric, values['value'], values['normalized_value'], values.get('normalized_factor'))
                 for metric, values in metrics.items()]
    metric_df = pd.DataFrame(metric_df,
                             columns=['Metric', 'Value', 'Normalized Value', 'Expected Value (in random net)'])
    st.subheader('Metrics')
    st.dataframe(metric_df)


def _chart_selectors(charts):
    st.subheader('Charts')
    scale = 'log' if st.checkbox('Display in Log Scale') else 'linear'
    chart_title = st.selectbox('Chart', list(charts.keys()))
    return chart_title, scale


def _encode_altair_chart(chart: alt.Chart, scale: str) -> alt.Chart:
    chart = chart.encode(alt.X('x', axis=alt.Axis(title=''), sort=None),
                         alt.Y('y', scale=alt.Scale(type=scale, base=10), axis=alt.Axis(title='')),
                         tooltip=['x', 'y'])
    return chart


def display_chart(charts):
    charts = {CHART_NAME_MAPPING.get(k, k): v for k, v in charts.items()}
    chart_title, scale = _chart_selectors(charts)
    chart = charts[chart_title]
    c = alt.Chart(pd.DataFrame(zip(chart['x'], chart['y']), columns=['x', 'y']),
                  title=chart_title,
                  height=500).mark_bar()
    st.altair_chart(_encode_altair_chart(c, scale), use_container_width=True)


def display_chart_with_mean_line(charts):
    charts = {CHART_NAME_MAPPING.get(k, k): v for k, v in charts.items()}
    chart_title, scale = _chart_selectors(charts)
    chart = charts[chart_title]
    altair_charts = []
    for index, y_value in enumerate(chart['ys']):
        c = alt.Chart(pd.DataFrame(zip(chart['x'], y_value), columns=['x', 'y']), title=chart_title, height=500).mark_line()
        altair_charts.append(_encode_altair_chart(c, scale))

    altair_chart = reduce(lambda a, b: a + b, altair_charts)
    mean_line = alt.Chart(pd.DataFrame(zip(chart['x'], chart['y']), columns=['x', 'y']), title=chart_title,
                          height=500).mark_line(color='black', strokeWidth=3)
    mean_line = _encode_altair_chart(mean_line, scale)
    st.altair_chart(altair_chart + mean_line, use_container_width=True)
