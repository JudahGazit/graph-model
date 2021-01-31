import networkx as nx
import numpy as np
import scipy.spatial
import streamlit as st
from matplotlib import pyplot as plt

from graph.datasets_loader import DatasetsLoader
from graph.metrics.costs.resistance_cost import ResistanceCost
from ui.resistance_centrality.data_loaders import load_brain
from ui.resistance_centrality.plotters import plot_fiber_length_dist, remove_node_from_dataset, brain_polygon, \
    plot_hist, plot_scatter, plot_brain_point_cloud, plot_3d_brain

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


def degree_utilization_in_percentiles(dataset, centrality, ax):
    degrees = np.count_nonzero(np.asarray(dataset.adjacency), axis=0).astype(np.float)
    utilization = []
    percentiles = np.linspace(0.1, 1, 10)
    for p in percentiles:
        allow_list = np.where(centrality <= np.quantile(centrality, p))[0]
        subgraph_dataset = remove_node_from_dataset(dataset, allow_list=allow_list)
        utilization_p = []
        for i in range(subgraph_dataset.number_of_nodes):
            current_degree = (subgraph_dataset.adjacency[i] > 0).sum()
            utilization_p.append(current_degree / degrees[subgraph_dataset.nx_graph.nodes[i]['original_label']])
        utilization.append(np.mean(utilization_p))
    plt.setp(ax, xticks=np.linspace(0, 1, 11), yticks=np.linspace(0, 1, 11),
             xlabel='percentile', ylabel='% of degree utilized',
             title='$\\frac{degree\ at\ percentile}{degree}$ at different percentiles')
    ax.bar(percentiles, utilization, width=0.05)
    ax.plot(percentiles, utilization, 'green')
    ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), 'red')


def modularity_efficiency_at_percentile(dataset, centrality, ax):
    x = np.linspace(0.1, 1, 20)
    modularity_scores = np.zeros_like(x)
    for i, p in enumerate(x):
        in_percentile = centrality <= np.quantile(centrality, p)
        is_same_component = np.array(np.meshgrid(in_percentile, in_percentile))
        is_same_component = (is_same_component[0, :, :] == is_same_component[1, :, :])
        modularity_mat = nx.modularity_matrix(dataset.nx_graph, range(dataset.number_of_nodes), weight=None)
        modularity_scores[i] = np.multiply(modularity_mat, is_same_component).sum() / (2 * dataset.number_of_edges)
    plt.setp(ax, ylim=(-abs(modularity_scores).max(), abs(modularity_scores).max()),
             xlabel='percentile', xticks=np.linspace(0, 1, 11), ylabel='modularity efficiency',
             title='modularity efficiency at percetiles')
    plt.setp([spline[1] for spline in ax.spines.items() if spline[0] not in ('left', 'bottom')], visible=False)
    ax.spines['bottom'].set_position('center')
    plt.bar(x, modularity_scores, 0.01)


def plot_percentile_modularity(dataset, centrality):
    fig = plt.figure(figsize=(20, 7))
    degree_utilization_in_percentiles(dataset, centrality, fig.add_subplot(1, 2, 1))
    modularity_efficiency_at_percentile(dataset, centrality, fig.add_subplot(1, 2, 2))
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
