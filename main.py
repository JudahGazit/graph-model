import streamlit as st

from ui.datasets import datasets
from ui.multiple_optimizations import multiple_optimizations
from ui.optimize import optimize
from ui.simulate import simulate

actions = {
    'Simulate': simulate,
    'Datasets': datasets,
    'Cost Optimize': optimize,
    'Previous Optimizations': multiple_optimizations
}

if __name__ == '__main__':
    st.set_page_config(page_title='Graph Models')
    selected_action = st.sidebar.selectbox('Action', list(actions.keys()))
    st.sidebar.title('Parameters')
    actions[selected_action]()

# @app.route('/api/optimizer/')
# def optimize():
#     num_nodes = request.args.get('nodes')
#     mean_degree = request.args.get('degree')
#     wiring_factor = request.args.get('wiring')
#     routing_factor = request.args.get('routing')
#     fuel_factor = request.args.get('fuel')
#     method = request.args.get('method')
#     num_edges = int(num_nodes) * float(mean_degree) / 2
#     optimizer = GraphOptimizer(int(num_nodes), int(num_edges), float(wiring_factor), float(routing_factor),
#                                float(fuel_factor), method)
#     graph_dataset = optimizer.optimize()
#     formatter = GraphFormatter(graph_dataset)
#     result = {
#         'edges': formatter.format_graph(),
#         'chart': formatter.format_chart(),
#         'metric': formatter.format_metrics()
#     }
#     return jsonify(result)
#
#
# @app.route('/api/optimizer/multiple/')
# def load_premade_optimizations():
#     loader = MultipleOptimizationsLoader()
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