import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.spatial
import scipy.stats
import streamlit as st
from matplotlib import animation
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture

from graph.datasets_loader import DatasetsLoader
from graph.graph_dataset import GraphDataset
from graph.graph_formatter import GraphFormatter
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.random_lattice import RandomLattice

np.set_printoptions(2)


def resistance_centrality(graph_dataset, node_resistance=0):
    resistance = ResistanceCost(graph_dataset)
    resistance.node_resistance = node_resistance
    omega = resistance.omega()
    omega[omega == 0] = np.nan
    hmean = omega.shape[0] / np.nansum(1 / omega, axis=0)
    hmean = np.asarray(hmean)[0]
    return hmean


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


def plot(graph_dataset: GraphDataset, node_colors=None, with_labels=True, mark_min=False):
    plt.figure(figsize=(17, 14))
    cmap = plt.get_cmap('jet')
    if node_colors is not None:
        vmin = node_colors.min()
        vmax = node_colors.max()
        if vmax - vmin > 1e-5:
            node_colors = (node_colors - vmin) / (vmax - vmin)
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            mappable._A = []
            plt.colorbar(mappable)
        if mark_min:
            position = graph_dataset.positions(node_colors.argmin())
            plt.scatter(position[0], position[1], c='black', linewidths=100, zorder=10)
    else:
        node_colors = None
    nx.draw_networkx(graph_dataset.nx_graph,
                     pos=[graph_dataset.positions(i) for i in range(graph_dataset.number_of_nodes)],
                     nodelist=range(graph_dataset.number_of_nodes),
                     node_color=node_colors, cmap=cmap, with_labels=with_labels, node_size=100)
    plt.axis('off')
    # plt.show()


@st.cache(allow_output_mutation=True)
def load_brain(brain_name):
    dataset_loader = DatasetsLoader()
    brain = dataset_loader.load(f'brain_nets_old/{brain_name}')
    positions = dataset_loader.load(f'brain_nets/{brain_name}')
    positions = [tuple([x[1] for x in sorted(positions.nx_graph.nodes[i].items())][:3]) for i in
                 range(brain.number_of_nodes)]
    brain.positions = lambda i: positions[i]
    return brain


def brain_polygon(brain, resistances, mappable):
    positions = np.array([brain.positions(i) for i in range(brain.number_of_nodes)])
    convex = scipy.spatial.ConvexHull(np.array(positions))
    nearest_plane = np.zeros(positions.shape[0])
    for i, point in enumerate(positions):
        distances = np.matmul(convex.equations[:, :3], point.transpose()) + convex.equations[:, 3]
        nearest_plane[i] = abs(distances).argmin()

    triangles = []
    for index, item in enumerate(convex.simplices):
        color_value = None
        if resistances is not None:
            color_value = resistances[nearest_plane == index]
            color_value = mappable.to_rgba(color_value.mean()) if len(color_value) > 0 else mappable.cmap(0)
        triangles.append((convex.points[item, 0], convex.points[item, 1], convex.points[item, 2], color_value))
    return triangles


def plot_3d_brain(brain, resistances=None, angle=45, return_animation=False, fig=None, ax=None):
    if ax is None or fig is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    ax.axis('off')
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                     norm=plt.Normalize(vmin=resistances.min(), vmax=resistances.max()))
    plt.colorbar(mappable)
    triangles = brain_polygon(brain, resistances, mappable)

    def init():
        for x, y, z, color in triangles:
            ax.plot_trisurf(x, y, z, color=color, )
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i * 5)
        return fig,

    if return_animation:
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360 // 5, interval=20, blit=True)
        return anim
    else:
        init()
        animate(angle)
        return fig


def color_fiber_length(dataset, x, centrality, method='max'):
    colors = np.zeros_like(x, dtype=np.float)
    num_nodes = []
    for i, single_x in enumerate(x):
        x_from, x_to = [float(item) for item in single_x[1:-1].split(', ')]
        in_range = (dataset.adjacency < x_to) & (dataset.adjacency >= x_from)
        nonzero = np.count_nonzero(in_range)
        num_nodes.append((in_range > 0).max(0).sum())
        if method == 'mean':
            colors[i] = np.multiply(in_range, centrality).sum() / nonzero if nonzero > 0 else np.nan
        if method == 'max':
            colors[i] = np.multiply(in_range, centrality).max()
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                     norm=plt.Normalize(vmin=float(np.nanmin(colors)), vmax=float(np.nanmax(colors))))
    colors = [mappable.to_rgba(color) for color in colors]
    return colors, mappable, num_nodes


def plot_fiber_length_dist(dataset, centrality=None, color=False, method='max', fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    chart = GraphFormatter(dataset).format_chart()['edge-length-dist-dbins']
    x = chart['x']
    colors = None
    ax.set_xticks([])
    if color:
        colors, mappable, num_nodes = color_fiber_length(dataset, x, centrality, method=method)
        plt.colorbar(mappable)
        for i, n in enumerate(num_nodes):
            ax.text(i, 0, n, horizontalalignment='center', fontsize=8)
    ax.set_title('fiber length distribution' + f', colored by {method} centrality of nodes' if color else '')
    ax.bar(range(len(x)), chart['y'], width=1, color=colors)
    return fig

def remove_node_from_dataset(dataset: GraphDataset, deny_list=None, allow_list=None):
    allow_list = allow_list if allow_list is not None else [node for node in range(dataset.number_of_nodes) if node not in deny_list]
    subgraph = dataset.nx_graph.subgraph(allow_list)
    subgraph = nx.convert_node_labels_to_integers(subgraph, label_attribute='original_label')
    subgraph_distances = np.mat([[brain.distances[subgraph.nodes[i]['original_label'], subgraph.nodes[j]['original_label']]
                                  for j in range(subgraph.number_of_nodes())] for i in range(subgraph.number_of_nodes())])
    subgraph_positions = lambda i: brain.positions(subgraph.nodes[i].original_label)
    subgraph_dataset = GraphDataset(subgraph, subgraph_distances, subgraph_positions)
    return subgraph_dataset


def load_city(city_name, osm=False):
    category = 'streets_osm' if osm else 'street_network'
    city = DatasetsLoader().load(f'{category}/{city_name}_GRAPH')
    positions = np.array(
        [(city.nx_graph.nodes[i]['X'], city.nx_graph.nodes[i]['Y']) for i in range(city.number_of_nodes)])
    city.positions = lambda i: positions[i]
    return city


def plot_scatter(x, y, c, title, xlabel, ylabel, colorbar=False, cmap=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, c=c, cmap=cmap)
    if colorbar:
        mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                         norm=plt.Normalize(vmin=c.min(), vmax=c.max()))
        plt.colorbar(mappable)
    return fig


def plot_hist(x, bins, title, xlabel, ylabel, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.hist(x, bins=bins)
    return fig


def plot_brain_point_cloud(dataset, centrality, angle, fig=None, ax=None):
    if ax is None or fig is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    points = np.array([dataset.positions(i) for i in range(dataset.number_of_nodes)])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=centrality)
    ax.view_init(elev=10., azim=angle)
    return fig


def _get_parameters():
    brain_options = DatasetsLoader().categories['brain_nets'].options
    brain_name = st.sidebar.selectbox('Brain Dataset', brain_options)
    node_resistance = st.sidebar.slider('Node Resistance', 0.0, 1.0e5, 0.0, 1.0)
    angle = st.slider('Angle', 0.0, 360.0, 45.0, 1.0)
    st.header(brain_name)
    return brain_name, node_resistance, angle


def brain_centrality():
    brain_name, node_resistance, angle = _get_parameters()

    brain = load_brain(brain_name)
    centrality = resistance_centrality(brain, node_resistance)

    fig = plt.figure(figsize=(20, 7))
    plot_3d_brain(brain, centrality, angle, False, fig, fig.add_subplot(1, 2, 1, projection='3d'))
    plot_brain_point_cloud(brain, centrality, angle, fig, fig.add_subplot(1, 2, 2, projection='3d'))
    st.pyplot(fig)

    degrees = np.count_nonzero(np.asarray(brain.adjacency), axis=0).astype(np.float)
    mean_edge_length = np.asarray(brain.adjacency).mean(0)
    max_edge_length = np.asarray(brain.adjacency).max(0)
    weighted_degrees = np.asarray(brain.adjacency).sum(0)
    per_80 = brain.distances.min() + 0.8 * (brain.distances.max() - brain.distances.min())
    color_80th_percent = (np.asarray(brain.adjacency) > per_80).any(0).astype(np.float)

    fig = plt.figure(figsize=(20, 7))
    plot_hist(centrality, 20, 'Resistance Centrality Histogram', 'resistance centrality', 'freq', fig=fig, ax=fig.add_subplot(1, 2, 1))
    plot_hist(np.reciprocal(centrality), 20, '(Resistance Centrality$)^{-1}$ Histogram',
              '(resistance centrality$)^{-1}$', 'freq', fig=fig, ax=fig.add_subplot(1, 2, 2))
    st.pyplot(fig)
    fig = plt.figure(figsize=(20, 7))
    plot_fiber_length_dist(brain, centrality, color=True, method='mean', fig=fig, ax=fig.add_subplot(1, 2, 1))
    plot_fiber_length_dist(brain, centrality, color=True, method='max', fig=fig, ax=fig.add_subplot(1, 2, 2))
    st.pyplot(fig)

    x_values_mapping = [
        ('Degree', degrees),
        ('$\\frac{1}{k_i} \\sum_j w_{ij}$', mean_edge_length),
        ('$\max_j w_{ij}$', max_edge_length),
        ('$\sum_j w_{ij}$', weighted_degrees),
    ]
    fig = plt.figure(figsize=(20, 15))
    for i, (x_label, x) in enumerate(x_values_mapping):
        plot_scatter(x, centrality, color_80th_percent, f'Resistance Centrality as function of {x_label}', x_label,
                     'resistance centrality', fig=fig, ax=fig.add_subplot(2, 2, i + 1))
    st.pyplot(fig)

    res = ResistanceCost(brain)
    res.node_resistance = node_resistance
    edge_centrality = brain.adjacency > 0
    value = res.cost().value
    edge_centrality = np.multiply(1 - edge_centrality, (value - res.costs_if_add())) +\
                      np.multiply(edge_centrality, (value - res.costs_if_remove()))
    fig = plt.figure(figsize=(20, 15))
    for do_abs in (False, True):
        for do_color in (False, True):
            edge_centrality_color = np.array(np.meshgrid(centrality, centrality)).max(0).flatten() if do_color else None
            edge_centrality = abs(edge_centrality) if do_abs else edge_centrality
            plot_scatter(np.asarray(brain.distances).flatten(), np.asarray(edge_centrality).flatten(), edge_centrality_color,
                         'Edge Centrality as function of Edge Length', 'Edge length', 'Edge Centrality', colorbar=do_color, cmap=plt.cm.jet,
                         fig=fig, ax=fig.add_subplot(2, 2, do_abs * 2 + do_color + 1))
    st.pyplot(fig)


if __name__ == '__main__':
    brain_name = 'Baboon3_FinalFinal2'
    node_resistance = 0
    brain = load_brain(brain_name)
    centrality = resistance_centrality(brain, node_resistance)
    positions = np.array([brain.positions(i) for i in range(brain.number_of_nodes)])
    convex = scipy.spatial.ConvexHull(np.array(positions))

    nearest_plane, plane_distances = np.zeros(positions.shape[0]), np.zeros(positions.shape[0])
    for i, point in enumerate(positions):
        distances = np.matmul(convex.equations[:, :3], point.transpose()) + convex.equations[:, 3]
        nearest_plane[i] = abs(distances).argmin()
        plane_distances[i] = abs(distances).min()

    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                     norm=plt.Normalize(vmin=centrality.min(), vmax=centrality.max()))

    triangles = brain_polygon(brain, centrality, mappable)


    top_percentile = np.quantile(centrality, 0.97)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for x, y, z, color in triangles:
        ax.plot_trisurf(x, y, z, color=color, alpha=0.5)
    ax.scatter(positions[(centrality > top_percentile) & (plane_distances >= 0)][:, 0],
               positions[(centrality > top_percentile) & (plane_distances >= 0)][:, 1],
               positions[(centrality > top_percentile) & (plane_distances >= 0)][:, 2],
               s=100, color='black', alpha=1)


    subgraph_dataset = remove_node_from_dataset(brain, allow_list=np.where(centrality < np.quantile(centrality, 0.4))[0])

    plot_fiber_length_dist(subgraph_dataset)

    plt.figure()
    layout = nx.spring_layout(subgraph_dataset.nx_graph, weight='weight')
    nx.draw_networkx(subgraph_dataset.nx_graph, layout)
    plt.show()

    rel = []
    for i in range(subgraph_dataset.number_of_nodes):
        rel_i = ((subgraph_dataset.adjacency[i] > 0).sum()) / (
            (brain.adjacency[subgraph_dataset.nx_graph.nodes[i]['original_label']] > 0).sum())
        rel.append(rel_i)
    rel = np.asarray(rel)


    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(sub_pos[:, 0], sub_pos[:, 1], sub_pos[:, 2],)
    plt.show()