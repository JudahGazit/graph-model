from graph.multiple_simulations_loader import MultipleOptimizationsLoader
from ui.utils import *


def format_parameters(params):
    params = [str(round(float(s), 3)) if '*' not in s else s for s in params]
    return ', '.join(params)

def _get_parameters(loader):
    nodes_edges = {}
    cost_types = list({option.cost_type for option in loader.options})
    selected_cost_type = st.sidebar.selectbox('Topology', cost_types)
    for option in loader.options:
        if option.cost_type == selected_cost_type:
            key = int(option.nodes), int(option.edges), option.remark
            value = (option.wiring_factor, option.routing_factor, option.fuel_factor)
            nodes_edges[key] = nodes_edges.get(key, []) + [value]
    selected_nodes_edges = st.sidebar.radio('(Nodes, Edges, REMARK)', sorted(nodes_edges),
                                            format_func=lambda v: ', '.join(map(str, v)))
    selected_params = st.sidebar.selectbox('(Wiring, Routing, Fuel)',
                                       sorted(nodes_edges[selected_nodes_edges], reverse=True,
                                              key=lambda x: [float(x[i]) for i in [1, 2, 0]]) + [('*', '*', '*')],
                                       format_func=format_parameters)
    selected_option = (selected_cost_type, ) + selected_nodes_edges + selected_params
    strategy = st.sidebar.select_slider('Strategy', ['best', 'mean'])
    return selected_option, strategy


def _recreate_graph(results):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(r['source'], r['target'], r['weight']) for r in results['edges']])
    return graph


def _display_graph(graph, topology):
    if topology == 'circular':
        display_graph_in_circle(graph)
    elif topology == 'lattice':
        display_graph_as_lattice(graph)


def multiple_optimizations():
    loader = MultipleOptimizationsLoader()
    selected_option, strategy = _get_parameters(loader)
    results = loader.load(strategy, *selected_option)
    graph = _recreate_graph(results)
    _display_graph(graph, selected_option[0])
    display_metrics(results['metric'])
    if strategy == 'best':
        display_charts(results['chart'])
    else:
        display_chart_with_mean_line(results['chart'])