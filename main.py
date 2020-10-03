from flask import Flask, jsonify, request
from flask_cors import CORS

from datasets_loader import DatasetsResultCache
from graph_formatter import GraphFormatter
from graph_simulator import GraphSimulator

app = Flask(__name__, static_url_path='', static_folder='build')
CORS(app)

dataset_result_cache = DatasetsResultCache()


@app.route('/api')
def simulate():
    leaves = request.args.get('leaves')
    A = request.args.get('A')
    B = request.args.get('B')
    alpha = request.args.get('alpha')
    beta = request.args.get('beta')
    graph = GraphSimulator(2 ** int(leaves), float(A), float(B), float(alpha), float(beta)).simulate()
    formatter = GraphFormatter(graph)
    result = {
        'edges': formatter.format_graph(),
        'chart': formatter.format_chart(),
        'metric': formatter.format_metrics()
    }
    return jsonify(result)


@app.route('/api/dataset/<dataset>')
def load_dataset(dataset):
    result = dataset_result_cache.get_results(dataset)
    return jsonify(result)


@app.route('/', defaults={'path': ''})
@app.route('/<path>')
def serve_html(path):
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run()
