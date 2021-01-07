import numpy as np
import math
from functools import reduce

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
from scipy.optimize import least_squares

CHART_NAME_MAPPING = {
    'edge-length-dist': 'Fiber Length Histogram',
    'edge-length-dist-dbins': 'Fiber Length Histogram (normalized by distances bin)',
    'degree-histogram': 'Node Degree Histogram',
    'node-path-len-dist': 'Unweighted Shortest Path Distance Histogram',
    'nodes-distance-dist': 'Weighted Shortest Path Distance Histogram',
    'degree-edge-distance-correlation': 'Correlation between NODE DEGREE to AVG EDGE WEIGHT',
    'degree-and-degree-of-neighbours': 'DEGREE and AVG DEGREE of neighbours',
    'triangles-hist': 'Triangles Histogram',
    'length-and-width': 'Fiber Length (x) and Fiber widths (y)'
}

def display_graph(graph: nx.Graph, node_method=None, edge_method=None, engine='neato'):
    for i, node in enumerate(graph.nodes):
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


def display_graph_as_lattice(graph: nx.Graph):
    n = int(math.sqrt(graph.number_of_nodes()))

    def node_method(graph, node, node_index):
        pos_x = node_index % n
        pos_y = int(node_index / n)
        graph.nodes[node]['pos'] = f'{pos_x},{pos_y}!'

    def edge_method(graph, edge, node_index):
        width = graph.edges[edge].get('width')
        if width:
            graph.edges[edge]['penwidth'] = round(graph.edges[edge]['width'], 3)

    display_graph(graph, node_method, edge_method=edge_method)

def display_metrics(metrics):
    metric_df = [(metric, values['value'], values['normalized_value'], values.get('optimal_value'), values.get('worst_value'))
                 for metric, values in metrics.items()]
    metric_df = pd.DataFrame(metric_df,
                             columns=['Metric', 'Value', 'Efficiency', 'Optimal Value', 'Worst Value'])
    st.subheader('Metrics')
    st.dataframe(metric_df)


def _chart_selectors(charts):
    st.subheader('Charts')
    scale = 'log' if st.checkbox('Display in Log Scale') else 'linear'
    return scale


def _encode_altair_chart(chart: alt.Chart, scale: str) -> alt.Chart:
    chart = chart.encode(alt.X('x', axis=alt.Axis(title=''), sort=None),
                         alt.Y('y', scale=alt.Scale(type=scale, base=10), axis=alt.Axis(title='')),
                         tooltip=['x', 'y'])
    return chart


def _extract_x_y_from_chart(chart):
    chart_xy = np.mat([
        [(float(x[1:-1].split(', ', 1)[0]) + float(x[1:-1].split(', ', 1)[1])) / 2 for x in chart['x']],
        chart['y']
    ]).transpose()
    chart_xy = chart_xy[np.all(chart_xy > 1e-10, axis=1).A1]
    X, Y = np.squeeze(np.asarray(chart_xy[:, 0])), np.squeeze(np.asarray(chart_xy[:, 1]))
    return X, Y


def exponent(A, B, lamda1, lamda2, x):
    return A * np.exp(- abs(lamda1) * x) + B * np.exp(abs(lamda2) * x)

def fit_exponential_function_to_chart(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    X = X[Y > 0]
    Y = Y[Y > 0]

    def loss_exp(args, x, y):
        return exponent(*args, x) - y

    res_exp = least_squares(loss_exp, [1, 0.01, 0.5, 0.5], args=(X, Y), ftol=1e-10)
    args = res_exp.x
    return args


CHART_HEIGHT = 500
CHART_WIDTH = 700

def display_exponential(chart_title, chart):
    if isinstance(chart['x'][0], str):
        X, Y = _extract_x_y_from_chart(chart)

        args = fit_exponential_function_to_chart(X, Y)
        X1 = np.linspace(0, len(X) - 1, 100)
        Y1 = exponent(*args, X1)
        exp_line = alt.Chart(pd.DataFrame(zip(X1, Y1), columns=['x', 'y']), title=chart_title,
                             height=CHART_HEIGHT, width=CHART_WIDTH).mark_line(color='red', strokeWidth=3)
        return exp_line


def display_charts(charts):
    charts = {CHART_NAME_MAPPING.get(k, k): v for k, v in charts.items()}
    scale = _chart_selectors(charts)
    for chart_title, chart in charts.items():
        c = alt.Chart(pd.DataFrame(zip(chart['x'], chart['y']), columns=['x', 'y']),
                      title=chart_title,
                      height=CHART_HEIGHT, width=CHART_WIDTH)
        c = getattr(c, f'mark_{chart.get("type", "bar")}')()

        line = _encode_altair_chart(c, scale)
        exponential_line = display_exponential(chart_title, chart)
        if exponential_line is not None:
            line += _encode_altair_chart(exponential_line, scale)
        st.altair_chart(line)

def display_chart_with_mean_line(charts):
    charts = {CHART_NAME_MAPPING.get(k, k): v for k, v in charts.items()}
    scale = _chart_selectors(charts)
    for chart_title, chart in charts.items():
        altair_charts = []
        for index, y_value in enumerate(chart['ys']):
            c = alt.Chart(pd.DataFrame(zip(chart['x'], y_value), columns=['x', 'y']), title=chart_title,
                          height=CHART_HEIGHT, width=CHART_WIDTH).mark_line()
            altair_charts.append(_encode_altair_chart(c, scale))

        altair_chart = reduce(lambda a, b: a + b, altair_charts)
        mean_line = alt.Chart(pd.DataFrame(zip(chart['x'], chart['y']), columns=['x', 'y']), title=chart_title,
                              height=CHART_HEIGHT, width=CHART_WIDTH).mark_line(color='black', strokeWidth=3)
        mean_line = _encode_altair_chart(mean_line, scale)
        line = altair_chart + mean_line
        exponential_line = display_exponential(chart_title, chart)
        if exponential_line is not None:
            line += _encode_altair_chart(exponential_line, scale)
        st.altair_chart(line)
