from graph.graph_formatter import GraphFormatter
from graph.graph_optimizer import GraphOptimizer
from ui.utils import *

last_parameters = None
current_graph = None


def _get_parameters():
    cost_type = st.sidebar.selectbox('Network Type', ['circular', 'lattice'])
    num_nodes = st.sidebar.slider('Number of Nodes', 10, 300) if cost_type == 'circular' else st.sidebar.select_slider(
        'Number of Nodes', [i ** 2 for i in range(2, 20)], 16)
    mean_degree = st.sidebar.slider('Mean Degree', 1.5, 10.0, 2.0, step=0.01)
    wiring_factor = st.sidebar.slider('Wiring Factor', -1.0, 1.0, 0.0, step=0.01)
    routing_factor = st.sidebar.slider('Routing Factor', -1.0, 1.0, 0.0, step=0.01)
    fuel_factor = st.sidebar.slider('Fuel Factor', -1.0, 1.0, 0.0, step=0.01)
    method = st.sidebar.select_slider('Method', ['minimize', 'maximize'], 'maximize')
    optimizer = st.sidebar.select_slider('Optimizer', ['annealing', 'genetic'])
    num_edges = int(num_nodes * mean_degree / 2)
    return num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, optimizer


def _display_graph_by_cost_type(graph, cost_type):
    if cost_type == 'circular':
        display_graph_in_circle(graph)
    elif cost_type == 'lattice':
        display_graph_as_lattice(graph)


def _initial_graph(num_nodes, cost_type):
    if cost_type == 'circular':
        return nx.circular_ladder_graph(num_nodes)
    if cost_type == 'lattice':
        return nx.grid_2d_graph(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)))


def optimize():
    global last_parameters, current_graph
    num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, optimizer_type = _get_parameters()
    optimizer = GraphOptimizer(num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, optimizer_type)
    start_simulation = st.sidebar.button('Start Optimization (might take a while)')

    if start_simulation:
        current_graph = optimizer.optimize()
        last_parameters = (num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, optimizer_type)

    if last_parameters == (num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, optimizer_type):
        formatter = GraphFormatter(current_graph, topology=cost_type)
        _display_graph_by_cost_type(current_graph.graph, cost_type)
        display_metrics(formatter.format_metrics())
        display_chart(formatter.format_chart())
    else:
        _display_graph_by_cost_type(_initial_graph(num_nodes, cost_type), cost_type)
