from graph.multiple_simulations_loader import MultipleOptimizationsLoader
from ui.utils import *


def _get_parameters(loader):
    nodes_edges = {}
    for option in loader.options:
        key = option.cost_type, int(option.nodes), int(option.edges)
        value = (option.wiring_factor, option.routing_factor, option.fuel_factor)
        nodes_edges[key] = nodes_edges.get(key, []) + [value]
    selected_nodes_edges = st.sidebar.radio('(Topology, Nodes, Edges)', sorted(nodes_edges),
                                            format_func=lambda v: ', '.join(map(str, v)))
    selected_params = st.sidebar.radio('(Wiring, Routing, Fuel)',
                                       sorted(nodes_edges[selected_nodes_edges], reverse=True),
                                       format_func=lambda v: ', '.join(map(lambda s: str(round(float(s), 3)), v)))
    selected_option = selected_nodes_edges + selected_params
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
        display_chart(results['chart'])
    else:
        display_chart_with_mean_line(results['chart'])