from collections import namedtuple
from dataclasses import dataclass

from pathos.multiprocessing import ProcessPool

from graph.graph_formatter import GraphFormatter
from graph.graph_optimizer import GraphOptimizer
from ui.utils import *
import numpy as np

last_parameters = None
current_graph = None
worst_values = None

CIRCULAR = 'circular'
LATTICE = 'torus'
COST_TYPES = [LATTICE]


@dataclass
class OptimizeParameters:
    num_nodes: int
    num_edges: int
    cost_type: str
    factors: dict
    wiring_factor: float = None
    routing_factor: float = None
    fuel_factor: float = None


def _get_factors(factor_names):
    factors = {}
    for factor in factor_names:
        factor_title = factor.title()
        if st.sidebar.checkbox(f'{factor_title}?', True):
            factors[factor] = st.sidebar.slider(f'{factor_title} Factor', -1.0, 1.0, 0.0, step=0.01)
        else:
            factors[factor] = None
    return factors


def _get_parameters():
    cost_type = st.sidebar.selectbox('Network Type', COST_TYPES)
    num_nodes = st.sidebar.slider('Number of Nodes', 10, 300) if cost_type == 'circular' else st.sidebar.select_slider(
        'Number of Nodes', [i ** 2 for i in range(2, 20)], 16)
    mean_degree = st.sidebar.slider('Mean Degree', 1.5, 40.0, 2.0, step=0.01)
    factors = _get_factors(['wiring', 'routing', 'fuel'])
    num_edges = int(num_nodes * mean_degree / 2)
    st.title(f'{cost_type.title()} - {int(np.sqrt(num_nodes))}x{int(np.sqrt(num_nodes))}')
    return OptimizeParameters(num_nodes, num_edges, cost_type, factors)


def _display_graph_by_cost_type(graph, cost_type):
    if cost_type == CIRCULAR:
        display_graph_in_circle(graph)
    elif cost_type == LATTICE:
        display_graph_as_lattice(graph)


def _initial_graph(num_nodes, cost_type):
    if cost_type == CIRCULAR:
        return nx.circular_ladder_graph(num_nodes)
    if cost_type == LATTICE:
        return nx.grid_2d_graph(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)))


# @st.cache
def optimize_graph(optimize_parameters: OptimizeParameters):
    optimizer = GraphOptimizer(optimize_parameters.num_nodes,
                               optimize_parameters.num_edges,
                               optimize_parameters.factors,
                               'maximize',
                               optimize_parameters.cost_type,
                               'annealing')
    return optimizer.optimize()


def optimize():
    global last_parameters, current_graph
    optimize_parameters = _get_parameters()
    start_simulation = st.sidebar.button('Start Optimization (might take a while)')

    if start_simulation:
        current_graph = optimize_graph(optimize_parameters)
        last_parameters = optimize_parameters

    if last_parameters == optimize_parameters:
        formatter = GraphFormatter(current_graph)
        _display_graph_by_cost_type(current_graph.nx_graph, optimize_parameters.cost_type)
        display_metrics(formatter.format_metrics())
        display_charts(formatter.format_chart())
    else:
        _display_graph_by_cost_type(_initial_graph(optimize_parameters.num_nodes, optimize_parameters.cost_type),
                                    optimize_parameters.cost_type)
