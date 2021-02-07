import itertools

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
import scipy.optimize
import streamlit as st
from matplotlib import pyplot as plt

from graph.datasets_loader import DatasetsLoader
from graph.graph_dataset import GraphDataset
from graph.graph_optimizer import GraphOptimizer
from graph.metrics.costs.fuel_cost import FuelCost
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.metrics.costs.routing_cost import RoutingCost
from graph.metrics.costs.wiring_cost import WiringCost
from ui.resistance_centrality.data_loaders import load_brain, remove_node_from_dataset, grid_n
from ui.resistance_centrality.plotters import plot_fiber_length_dist, brain_polygon, \
    plot_hist, plot_scatter, plot_brain_point_cloud, plot_3d_brain, plot

np.set_printoptions(2)

def resistance_centrality(graph_dataset, node_resistance=0):
    resistance = ResistanceCost(graph_dataset)
    resistance.node_resistance = node_resistance
    omega = resistance.omega()
    omega[omega == 0] = np.nan
    hmean = omega.shape[0] / np.nansum(1 / omega, axis=0)
    hmean = np.asarray(hmean)[0]
    return hmean


def edge_resistance_centrality(graph_dataset, node_resistance=0):
    resistance = ResistanceCost(graph_dataset)
    resistance.node_resistance = node_resistance
    edge_centrality = graph_dataset.adjacency > 0
    value = resistance.cost().value
    edge_centrality = np.multiply(1 - edge_centrality, (value - resistance.costs_if_add())) + \
                      np.multiply(edge_centrality, (value - resistance.costs_if_remove()))
    return edge_centrality


def _get_parameters():
    brain_options = DatasetsLoader().categories['brain_nets'].options
    brain_name = st.sidebar.selectbox('Brain Dataset', brain_options)
    node_resistance = st.sidebar.slider('Node Resistance', 0.0, 1.0e5, 0.0, 1.0)
    angle = st.slider('Angle', 0.0, 360.0, 45.0, 1.0)
    st.header(brain_name)
    return brain_name, node_resistance, angle


def plot_3d_brains(brain, centrality, angle):
    fig = plt.figure(figsize=(20, 7))
    plot_3d_brain(brain, centrality, angle, False, fig, fig.add_subplot(1, 2, 1, projection='3d'))
    plot_brain_point_cloud(brain, centrality, angle, fig, fig.add_subplot(1, 2, 2, projection='3d'))
    st.pyplot(fig)


def plot_length_edge_centrality_dependency(brain, centrality, edge_centrality):
    edge_centrality_color = np.array(np.meshgrid(centrality, centrality)).max(0).flatten()
    x, y = np.asarray(brain.distances).flatten(), np.asarray(edge_centrality).flatten()
    fig = plt.figure(figsize=(20, 15))
    for do_abs in (False, True):
        for do_color in (False, True):
            xlabel, ylabel = 'Edge length', "$|$Edge Centrality$|$" if do_abs else "Edge Centrality"
            plot_scatter(x, abs(y) if do_abs else y,
                         edge_centrality_color if do_color else None,
                         f'{ylabel} as function of {xlabel}', xlabel, ylabel,
                         colorbar=do_color, cmap=plt.cm.jet,
                         fig=fig, ax=fig.add_subplot(2, 2, do_abs * 2 + do_color + 1))
    st.pyplot(fig)


def plot_degree_centrality_dependencies(brain, centrality):
    degrees = np.count_nonzero(np.asarray(brain.adjacency), axis=0).astype(np.float)
    mean_edge_length = np.asarray(brain.adjacency).mean(0)
    max_edge_length = np.asarray(brain.adjacency).max(0)
    weighted_degrees = np.asarray(brain.adjacency).sum(0)
    per_80 = brain.distances.min() + 0.8 * (brain.distances.max() - brain.distances.min())
    color_80th_percent = (np.asarray(brain.adjacency) > per_80).any(0).astype(np.float)
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


def plot_fiber_length_histograms(brain, centrality):
    fig = plt.figure(figsize=(20, 7))
    plot_fiber_length_dist(brain, centrality, color=True, method='mean', fig=fig, ax=fig.add_subplot(1, 2, 1))
    plot_fiber_length_dist(brain, centrality, color=True, method='max', fig=fig, ax=fig.add_subplot(1, 2, 2))
    st.pyplot(fig)


def plot_centrality_histograms(centrality):
    fig = plt.figure(figsize=(20, 7))
    plot_hist(centrality, 20, 'Resistance Centrality Histogram', 'resistance centrality', 'freq', fig=fig,
              ax=fig.add_subplot(1, 2, 1))
    plot_hist(np.reciprocal(centrality), 20, '(Resistance Centrality$)^{-1}$ Histogram',
              '(resistance centrality$)^{-1}$', 'freq', fig=fig, ax=fig.add_subplot(1, 2, 2))
    st.pyplot(fig)


def plot_fiber_length_at_centrality_percentile(dataset, centrality, percentile, above_or_below, fig, ax):
    sub_dataset = None
    if percentile < 1:
        if above_or_below == 'below':
            allow_list = np.where(centrality <= np.quantile(centrality, percentile))[0]
        else:
            allow_list = np.where(centrality > np.quantile(centrality, percentile))[0]
        sub_dataset = remove_node_from_dataset(dataset, allow_list=allow_list)

    if sub_dataset is not None and sub_dataset.is_connected:
        plot_fiber_length_dist(sub_dataset, fig=fig, ax=ax)
    else:
        ax.text(0.51, 0.5, 'not\nconnected', horizontalalignment='center')


def plot_fiber_length_at_centrality_percentiles(dataset, centrality):
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.setp(ax, xticks=np.linspace(0, 1, 11), yticks=[], xlim=(0, 1), ylim=(-1, 1))
    plt.setp([spine[1] for spine in ax.spines.items() if spine[0] != 'bottom'], visible=False)
    ax.spines['bottom'].set_position('center')
    ax.text(-0.1, 0.1, 'above percentile $\\uparrow$')
    ax.text(-0.1, -0.1, 'under percentile $\\downarrow$', )
    for i, p in enumerate(np.linspace(0.1, 1, 10)):
        ax_top = ax.inset_axes((p, 0, 0.045, 0.5), transform=ax.transData, zorder=1)
        ax_bottom = ax.inset_axes((p - 0.045, -0.5, 0.045, 0.5), transform=ax.transData, zorder=1)
        plt.setp((ax_top, ax_bottom), xticks=[], yticks=[])
        plot_fiber_length_at_centrality_percentile(dataset, centrality, p, 'below', fig, ax_bottom)
        plot_fiber_length_at_centrality_percentile(dataset, centrality, p, 'above', fig, ax_top)
    st.pyplot(fig)


def percentiles_bar_chart(dataset, centrality, method, ax,
                          error_lines=False, value_lines=False, x_shift=0, percentiles=np.linspace(0.1, 1, 10),
                          n_random=100, width=0.02):
    percentile_results = []
    random_means = np.zeros_like(percentiles)
    random_error_ranges = np.zeros((2, percentiles.shape[0]))
    n_random = (error_lines and n_random)
    for i, p in enumerate(percentiles):
        percentile_results.append(method(dataset, centrality <= np.quantile(centrality, p)))
        random_values = [method(dataset, np.random.random(dataset.number_of_nodes) <= p) for _ in range(n_random)]
        if n_random:
            random_means[i] = np.mean(random_values)
            random_error_ranges[0, i] = random_means[i] - np.quantile(random_values, 0.25)
            random_error_ranges[1, i] = np.quantile(random_values, 0.75) - random_means[i]
    plt.setp(ax, xticks=np.linspace(0, 1, 11), xlabel='percentile')
    ax.bar(percentiles - x_shift, percentile_results, width=width, label='resistance centrality percentile')
    if value_lines:
        ax.plot(percentiles - x_shift, percentile_results, 'green')
    if error_lines:
        ax.errorbar(percentiles + x_shift + width, random_means,
                    yerr=random_error_ranges,
                    xerr=width / 2, color='orange', label='random')
        ax.legend()
    return percentile_results


def percentile_subgraph(func):
    def inner(dataset, subgraph_nodes):
        allow_list = np.where(subgraph_nodes)[0]
        subgraph = remove_node_from_dataset(dataset, allow_list=allow_list)
        return func(subgraph)
    return inner


def degree_utilization_in_percentiles(dataset, centrality, ax):
    degrees = (np.asarray(dataset.adjacency) > 0).sum(0)

    @percentile_subgraph
    def method(subgraph_dataset):
        current_degrees = (np.asarray(subgraph_dataset.adjacency) > 0).sum(0)
        nodes_original_labels = [subgraph_dataset.nx_graph.nodes[i]['original_label'] for i in range(subgraph_dataset.number_of_nodes)]
        original_degrees = degrees[nodes_original_labels]
        return np.mean(current_degrees / original_degrees)

    percentiles_bar_chart(dataset, centrality, method, ax, value_lines=True, width=0.05)
    plt.setp(ax, yticks=np.linspace(0, 1, 11), ylabel='% of degree utilized',
             title='$\\mathbb{E}[\\frac{degree\ at\ percentile}{degree}]$ at different percentiles')
    ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), 'red')


def modularity_efficiency_at_percentile(dataset, centrality, ax):
    def method(dataset, in_percentile):
        for i in range(dataset.number_of_nodes):
            is_same_component = np.array(np.meshgrid(in_percentile, in_percentile))
            is_same_component = (is_same_component[0, :, :] == is_same_component[1, :, :])
            modularity_mat = nx.modularity_matrix(dataset.nx_graph, range(dataset.number_of_nodes), weight=None)
        modularity = np.multiply(modularity_mat, is_same_component).sum() / (2 * dataset.number_of_edges)
        return modularity

    percentile_results = percentiles_bar_chart(dataset, centrality, method, ax, percentiles=np.linspace(0.1, 1.0, 20))
    plt.setp(ax, ylim=(-np.abs(percentile_results).max(), np.abs(percentile_results).max()),
             ylabel='modularity efficiency', title='modularity efficiency at percetiles')
    plt.setp([spline[1] for spline in ax.spines.items() if spline[0] not in ('left', 'bottom')], visible=False)
    ax.spines['bottom'].set_position('center')


def travel_costs_at_percentiles(dataset, centrality, below_or_above='below', ax=None):
    costs = []
    percentiles = np.linspace(0.1, 1.0, 10) if below_or_above == 'below' else np.linspace(0.1, 0.9, 9)
    for i, p in enumerate(percentiles):
        allow_list = np.where(centrality <= np.quantile(centrality, p)
                              if below_or_above == 'below' else centrality > np.quantile(centrality, p))[0]
        brain_at_percentile = remove_node_from_dataset(dataset, allow_list=allow_list)
        costs.append((WiringCost(brain_at_percentile).cost().normalized_value,
                      RoutingCost(brain_at_percentile).cost().normalized_value))
    costs = np.array(costs)
    plt.setp(ax, ylim=(0.5, 1), xticks=percentiles,
             xlabel='percentiles', ylabel=f'normalized cost', title=f'normalized routing & wiring costs {below_or_above} percentiles')
    ax.bar(percentiles - 0.01, costs[:, 0], 0.02, label='wiring')
    ax.bar(percentiles + 0.01, costs[:, 1], 0.02, label='routing')
    ax.legend()


def relative_convex_volume_at_percentiles(dataset, centrality, ax):
    original_volume = ConvexHull([dataset.positions(i) for i in range(dataset.number_of_nodes)]).volume

    @percentile_subgraph
    def method(subdataset):
        percentile_volume = ConvexHull([subdataset.positions(i) for i in range(subdataset.number_of_nodes)]).volume
        return percentile_volume / original_volume

    plt.setp(ax, ylim=(0, 1), ylabel='% of total volume', title=f'% of total volume in percentiles & in random graphs')
    percentiles_bar_chart(dataset, centrality, method, ax, error_lines=True)


def number_of_edges_at_percentiles(dataset, centrality, ax):
    @percentile_subgraph
    def method(subgraph_dataset):
        return np.mean(subgraph_dataset.number_of_edges / dataset.number_of_edges)

    percentiles_bar_chart(dataset, centrality, method, ax, value_lines=True, width=0.05)
    plt.setp(ax, yticks=np.linspace(0, 1, 11), ylabel='% of total edges',
             title='$\\frac{num\ edges\ at\ percentile}{num\ edges}$ at different percentiles')
    ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 2, 'red')


def core_periphery_cost_at_percentiles(dataset, centrality, ax):
    def method(dataset, core):
        delta = np.asarray(np.meshgrid(core, core)).max(0)
        return np.multiply(dataset.adjacency, delta).sum()

    percentiles_bar_chart(dataset, centrality, method, ax, error_lines=True)
    plt.setp(ax, ylabel='core-periphery efficiency', title=f'core-periphery efficiency in percentiles & in random graphs')


def plot_percentile_modularity(dataset, centrality):
    n_rows = 4
    fig = plt.figure(figsize=(20, 7 * n_rows))
    degree_utilization_in_percentiles(dataset, centrality, fig.add_subplot(n_rows, 2, 1))
    number_of_edges_at_percentiles(dataset, centrality, fig.add_subplot(n_rows, 2, 2))
    modularity_efficiency_at_percentile(dataset, centrality, fig.add_subplot(n_rows, 2, 3))
    relative_convex_volume_at_percentiles(dataset, centrality, fig.add_subplot(n_rows, 2, 4))
    travel_costs_at_percentiles(dataset, centrality, 'below', fig.add_subplot(n_rows, 2, 5))
    travel_costs_at_percentiles(dataset, centrality, 'above', fig.add_subplot(n_rows, 2, 6))
    core_periphery_cost_at_percentiles(dataset, centrality, fig.add_subplot(n_rows, 2, 7))
    st.pyplot(fig)


def brain_centrality():
    brain_name, node_resistance, angle = _get_parameters()

    brain = load_brain(brain_name)
    centrality = resistance_centrality(brain, node_resistance)
    edge_centrality = edge_resistance_centrality(brain, node_resistance)

    plot_3d_brains(brain, centrality, angle)
    plot_centrality_histograms(centrality)
    plot_fiber_length_histograms(brain, centrality)
    plot_degree_centrality_dependencies(brain, centrality)
    plot_length_edge_centrality_dependency(brain, centrality, edge_centrality)
    plot_fiber_length_at_centrality_percentiles(brain, centrality)
    plot_percentile_modularity(brain, centrality)


if __name__ == '__main__':
    brain_name = 'Baboon3_FinalFinal2'
    node_resistance = 0
    dataset = load_brain(brain_name)
    centrality = resistance_centrality(dataset, node_resistance)
    positions = np.array([dataset.positions(i) for i in range(dataset.number_of_nodes)])
    convex = ConvexHull(np.array(positions))

    nearest_plane, plane_distances = np.zeros(positions.shape[0]), np.zeros(positions.shape[0])
    for i, point in enumerate(positions):
        distances = np.matmul(convex.equations[:, :3], point.transpose()) + convex.equations[:, 3]
        nearest_plane[i] = abs(distances).argmin()
        plane_distances[i] = abs(distances).min()

    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                     norm=plt.Normalize(vmin=centrality.min(), vmax=centrality.max()))

    triangles = brain_polygon(dataset, centrality, mappable)

    minibrain = remove_node_from_dataset(dataset, allow_list=np.where(centrality <= np.quantile(centrality, 0.4))[0])
    minibrain_centrality = resistance_centrality(minibrain)
    # plot_brain_point_cloud(dataset, centrality <= np.quantile(centrality, 0.4), 45)

    # subgrid = remove_node_from_dataset(dataset, allow_list=np.random.permutation(range(dataset.number_of_nodes))[:36])
    grid = grid_n(25)
    subgrid = remove_node_from_dataset(grid, allow_list=np.random.permutation(range(grid.number_of_nodes))[:9 ** 2])
    optimized = GraphOptimizer(subgrid.number_of_nodes,
                               subgrid.number_of_nodes * 6,
                               dict(fuel=1e-1, wiring=1e-9, routing=1),
                               'minimize',
                               'torus',
                               'annealing')
    optimized.graph_cost.distance_matrix = subgrid.distances
    optimized = optimized.optimize()
    adjacency = np.multiply(optimized.adjacency > 0, subgrid.distances)
    subgrid = GraphDataset(nx.from_numpy_matrix(adjacency), subgrid.distances, subgrid.positions, adjacency=adjacency)
    plot_fiber_length_dist(subgrid)
    plot(subgrid)

    tags = centrality <= np.quantile(centrality, 0.1)
    k = 3
    nearest = dataset.distances.argsort(1)[:, 1:k+1]
    predicts = tags[nearest].sum(1) >= (k // 2 + 1)
    score = (tags == predicts).mean()
    print(score)


    grid = grid_n(25)
    positions_g = np.array([grid.positions(i) for i in range(dataset.number_of_nodes)])
    number_of_hubs = int(0.2 * grid.number_of_nodes)
    p = 0.05
    random_hubs = np.random.permutation(range(grid.number_of_nodes))[:number_of_hubs]
    random_hub_edges = np.random.random((number_of_hubs, number_of_hubs)) < p
    random_hub_edges[np.triu_indices_from(random_hub_edges)] = 0
    random_hub_edges = random_hub_edges + random_hub_edges.transpose()
    random_hub_edges = random_hubs[np.argwhere(random_hub_edges)]
    grid_h = grid.nx_graph.copy()
    grid_h.add_weighted_edges_from([(u, v, grid.distances[u, v]) for u, v in random_hub_edges])
    grid_h = GraphDataset(grid_h, grid.distances, grid.positions)
    plot_fiber_length_dist(grid_h)

    plot_brain_point_cloud(dataset, centrality < np.quantile(centrality, 0.2), 45)
    # plt.plot(x, x, 'red')
    # plt.bar

    cliques = list(nx.find_cliques(dataset.nx_graph))

    mod = nx.convert_node_labels_to_integers(
        nx.Graph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (2, 7), (2, 8), (3, 4), (3, 8), (3, 9), (4, 5), (4, 9),
                  (8, 9), (8, 12), (10, 11), (10, 12), (11, 12)]))
    plt.figure()
    nx.draw_networkx(mod)
    plt.show()
    expected = [0, 1, 2, 3, 6, 7]
    components = np.array([1 if i in expected else 0 for i in range(mod.number_of_nodes())])
    adj = nx.to_numpy_array(mod, range(mod.number_of_nodes()))

    def score_center_module(adj, module):
        module = module.round()
        delta = np.asarray(np.meshgrid(module, module)).max(0)
        score = np.multiply(adj, delta).mean()
        return score


    minimum = scipy.optimize.brute(lambda module: -score_center_module(adj, module),
                                   [(0, 1) for i in range(mod.number_of_nodes())], Ns=2, disp=True)
    np.where(minimum.round())[0]
    sorted(results.items(), key=lambda x: (x[1], x[0].split('-')))