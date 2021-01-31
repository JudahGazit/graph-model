import networkx as nx
import numpy as np
import streamlit as st

from graph.datasets_loader import DatasetsLoader
from graph.graph_dataset import GraphDataset
from graph.random_lattice import RandomLattice


def grid_n(n):
    graph_dataset = RandomLattice(n ** 2, (n ** 2) * 4).randomize()
    grid_edges = np.argwhere(graph_dataset.distances == 1)
    grid = nx.Graph()
    grid.add_nodes_from(range(n ** 2))
    grid.add_weighted_edges_from([(u, v, 1) for u, v in grid_edges])
    return GraphDataset(grid, graph_dataset.distances, graph_dataset.positions)


def tree_n(n):
    tree = nx.balanced_tree(3, n)
    positions = nx.planar_layout(tree)
    distances = 1 - np.eye(tree.number_of_nodes())
    return GraphDataset(tree, distances=distances, positions=lambda i: positions[i])


@st.cache(allow_output_mutation=True)
def load_brain(brain_name):
    dataset_loader = DatasetsLoader()
    brain = dataset_loader.load(f'brain_nets_old/{brain_name}')
    positions = dataset_loader.load(f'brain_nets/{brain_name}')
    positions = [tuple([x[1] for x in sorted(positions.nx_graph.nodes[i].items())][:3]) for i in
                 range(brain.number_of_nodes)]
    brain.positions = lambda i: positions[i]
    return brain


def load_city(city_name, osm=False):
    category = 'streets_osm' if osm else 'street_network'
    city = DatasetsLoader().load(f'{category}/{city_name}_GRAPH')
    positions = np.array(
        [(city.nx_graph.nodes[i]['X'], city.nx_graph.nodes[i]['Y']) for i in range(city.number_of_nodes)])
    city.positions = lambda i: positions[i]
    return city