from graph.graph_formatter import GraphFormatter
from graph.random_lattice import RandomLattice
from ui.utils import *


current_parameters = None
current_graph = None


def _get_parameters():
    num_nodes = st.sidebar.select_slider('Number of Nodes', [i ** 2 for i in range(2, 20)], 16)
    mean_degree = st.sidebar.slider('Mean Degree', 1.5, 40.0, 2.0, 0.01)
    num_edges = int(num_nodes * mean_degree / 2)
    return num_nodes, num_edges


def simulate_lattice():
    global current_graph, current_parameters
    nodes, edges = _get_parameters()
    start_simulate = st.sidebar.button('Simulate')

    if start_simulate:
        current_graph = RandomLattice(nodes, edges).randomize()
        current_parameters = (nodes, edges)

    if current_parameters == (nodes, edges):
        formatter = GraphFormatter(current_graph, topology='lattice')
        display_graph_as_lattice(current_graph.graph)
        display_metrics(formatter.format_metrics())
        display_chart(formatter.format_chart())
    else:
        n = int(math.sqrt(nodes))
        display_graph_as_lattice(nx.grid_graph((n, n)))
