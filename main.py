import logging

from flask import Flask, jsonify, request
from flask.logging import default_handler
from flask_cors import CORS

from graph.datasets_loader import DatasetsResultCache
from graph.graph_formatter import GraphFormatter
from graph.graph_optimizer import GraphOptimizer
from graph.graph_simulator import GraphSimulator

app = Flask(__name__, static_url_path='', static_folder='build')
CORS(app)

dataset_result_cache = DatasetsResultCache()
app.logger.setLevel(logging.INFO)

@app.route('/api')
def simulate():
    leaves = request.args.get('leaves')
    A = request.args.get('A')
    B = request.args.get('B')
    alpha = request.args.get('alpha')
    beta = request.args.get('beta')
    graph_dataset = GraphSimulator(2 ** int(leaves), float(A), float(B), float(alpha), float(beta)).simulate()
    formatter = GraphFormatter(graph_dataset)
    result = {
        'edges': formatter.format_graph(),
        'chart': formatter.format_chart(),
        'metric': formatter.format_metrics()
    }
    return jsonify(result)


@app.route('/api/dataset/')
def load_dataset():
    category = request.args.get('category')
    dataset = request.args.get('dataset')
    result = dataset_result_cache.get_results('/'.join([category, dataset]))
    return jsonify(result)


@app.route('/api/datasets')
def load_datasets():
    data = dataset_result_cache.options()
    return jsonify(data)


@app.route('/api/optimizer/')
def optimize():
    num_nodes = request.args.get('nodes')
    mean_degree = request.args.get('degree')
    wiring_factor = request.args.get('wiring')
    routing_factor = request.args.get('routing')
    fuel_factor = request.args.get('fuel')
    method = request.args.get('method')
    num_edges = int(num_nodes) * float(mean_degree) / 2
    optimizer = GraphOptimizer(int(num_nodes), int(num_edges), float(wiring_factor), float(routing_factor), float(fuel_factor), method)
    graph_dataset = optimizer.optimize()
    formatter = GraphFormatter(graph_dataset)
    result = {
        'edges': formatter.format_graph(),
        'chart': formatter.format_chart(),
        'metric': formatter.format_metrics()
    }
    return jsonify(result)


@app.route('/', defaults={'path': ''})
@app.route('/<path>')
def serve_html(path):
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run()
