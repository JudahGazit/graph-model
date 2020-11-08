from graph.multiple_simulations_loader import MultipleOptimizationsLoader
from ui.utils import *
import re
# loader = MultipleOptimizationsLoader()
#     num_nodes = request.args.get('nodes')
#     num_edges = request.args.get('edges')
#     wiring_factor = request.args.get('wiring')
#     routing_factor = request.args.get('routing')
#     fuel_factor = request.args.get('fuel')
#     if num_nodes:
#         data = loader.load(num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor)
#     else:
#         data = loader.options
#     return jsonify(data)


def multiple_optimizations():
    loader = MultipleOptimizationsLoader()
    st.sidebar.text('(Nodes, Edges, Wiring, Routing, Fuel)')
    selected_option = st.sidebar.radio('Parameters', options=[
        f'({o.nodes}, {o.edges}, {o.wiring_factor}, {o.routing_factor}, {o.fuel_factor})'
        for o in loader.options
    ])
    selected_option = selected_option[1:-1]
    results = loader.load(*selected_option.split(', '))
    display_metrics(results['metric'])
    display_chart_with_mean_line(results['chart'])