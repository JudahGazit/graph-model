import streamlit as st
import networkx as nx

from graph.graph_formatter import GraphFormatter
from graph.graph_optimizer import GraphOptimizer
from ui.utils import *

last_parameters = None
current_graph = None


def _get_parameters():
    global last_parameters
    num_nodes = st.sidebar.slider('Number of Nodes', 10, 300)
    mean_degree = st.sidebar.slider('Mean Degree', 1.0, 10.0, step=0.01)
    wiring_factor = st.sidebar.slider('Wiring Factor', 0.0, 1.0, step=0.01)
    routing_factor = st.sidebar.slider('Routing Factor', 0.0, 1.0, step=0.01)
    fuel_factor = st.sidebar.slider('Fuel Factor', 0.0, 1.0, step=0.01)
    method = st.sidebar.select_slider('Method', ['minimize', 'maximize'])
    num_edges = int(num_nodes * mean_degree / 2)
    return num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method


def optimize():
    global last_parameters, current_graph
    num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method = _get_parameters()
    optimizer = GraphOptimizer(num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method)
    start_simulation = st.sidebar.button('Start Optimization (might take a while)')

    if start_simulation and last_parameters != (
    num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method):
        current_graph = optimizer.optimize()
        last_parameters = (num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method)

    if last_parameters == (num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method):
        formatter = GraphFormatter(current_graph)
        display_graph_in_circle(current_graph.graph)
        display_metrics(formatter.format_metrics())
        display_chart(formatter.format_chart())
    else:
        display_graph_in_circle(nx.circular_ladder_graph(num_nodes))
