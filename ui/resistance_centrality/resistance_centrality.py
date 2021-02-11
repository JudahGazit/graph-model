import matplotlib.patches
import networkx as nx
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture

from graph.datasets_loader import DatasetsLoader
from graph.graph_dataset import GraphDataset
from graph.metrics.costs.resistance_cost import ResistanceCost
from graph.metrics.costs.routing_cost import RoutingCost
from graph.metrics.costs.wiring_cost import WiringCost
from ui.resistance_centrality.data_loaders import load_brain, remove_node_from_dataset, random_model
from ui.resistance_centrality.plotters import *

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
    brain_options = ['Random Model'] + DatasetsLoader().categories['brain_nets'].options
    brain_name = st.sidebar.selectbox('Brain Dataset', brain_options)
    node_resistance = st.sidebar.slider('Node Resistance', 0.0, 1.0e5, 0.0, 1.0)
    angle = st.slider('Angle', 0.0, 360.0, 45.0, 1.0)
    st.header(brain_name)
    return brain_name, node_resistance, angle


def plot_3d_brains(brain, centrality, angle):
    fig = plt.figure(figsize=(20, 7))
    try:
        plot_3d_brain(brain, centrality, angle, False, fig, fig.add_subplot(1, 2, 1, projection='3d'))
        plot_brain_point_cloud(brain, centrality, angle, fig, fig.add_subplot(1, 2, 2, projection='3d'))
    except:
        plot(brain, centrality, fig=fig, ax=fig.add_subplot(1, 2, 1))
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
    percentiles = np.linspace(0.1, 0.9, 9)
    plt.setp(ax, xticks=percentiles, yticks=[], xlim=(0, 1), ylim=(-1, 1), title='Fiber Length at Percentiles')
    plt.setp([spine[1] for spine in ax.spines.items() if spine[0] != 'bottom'], visible=False)
    ax.spines['bottom'].set_position('center')
    ax.text(0.04, 0.3, 'PERIPHERY', ha='center')
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 0.7, color=(0, 0, 0.7, 0.2)))
    ax.text(0.04, -0.4, 'CORE', ha='center')
    ax.add_patch(matplotlib.patches.Rectangle((0, -0.7), 1, 0.7, color=(0.7, 0, 0, 0.2)))
    ax.text(-0.03, -0.02, 'Percentile:', ha='center')
    for i, p in enumerate(percentiles):
        ax_top = ax.inset_axes((p - 0.045 / 2, 0.1, 0.045, 0.5), transform=ax.transData, zorder=1)
        ax_bottom = ax.inset_axes((p - 0.045 / 2, -0.65, 0.045, 0.5), transform=ax.transData, zorder=1)
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

    percentiles_bar_chart(dataset, centrality, method, ax, value_lines=True)
    plt.setp(ax, yticks=np.linspace(0, 1, 11), ylabel='% of degree utilized',
             title='avg. $[\\frac{Core\ Degree}{degree}]$ at different percentiles')
    ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), 'red')


def modularity_efficiency_at_percentile(dataset, centrality, ax):
    def method(dataset, in_percentile):
        modularity_mat = nx.modularity_matrix(dataset.nx_graph, range(dataset.number_of_nodes), weight=None)
        is_same_component = np.array(np.meshgrid(in_percentile, in_percentile))
        is_same_component = (is_same_component[0, :, :] == is_same_component[1, :, :])
        modularity = np.multiply(modularity_mat, is_same_component).sum() / (2 * dataset.number_of_edges)
        return modularity

    percentile_results = percentiles_bar_chart(dataset, centrality, method, ax, percentiles=np.linspace(0.1, 1.0, 20))
    plt.setp(ax, ylim=(-np.abs(percentile_results).max(), np.abs(percentile_results).max()),
             ylabel='modularity efficiency', title='modularity efficiency at percetiles')
    plt.setp([spline[1] for spline in ax.spines.items() if spline[0] not in ('left', 'bottom')], visible=False)
    ax.spines['bottom'].set_position('center')


def travel_costs_at_percentiles(dataset, centrality, ax=None):
    percentiles = np.linspace(0.1, 0.9, 9)
    wiring = np.empty((percentiles.shape[0], 2))
    routing = np.empty((percentiles.shape[0], 2))
    for i, p in enumerate(percentiles):
        allow_list = np.where(centrality <= np.quantile(centrality, p))[0]
        brain_below_percentile = remove_node_from_dataset(dataset, allow_list=allow_list)
        brain_above_percentile = remove_node_from_dataset(dataset, deny_list=allow_list)
        wiring[i] = (WiringCost(brain_below_percentile).cost().normalized_value, WiringCost(brain_above_percentile).cost().normalized_value)
        routing[i] = (RoutingCost(brain_below_percentile).cost().normalized_value, RoutingCost(brain_above_percentile).cost().normalized_value)

    xticks = np.linspace(-1, 1, 5)
    plt.setp(ax, xlim=(-1, 1), yticks=percentiles, ylim=(0, 1), ylabel='percentile cutoff',
             xticks=xticks, xticklabels=abs(xticks),
             xlabel='normalized efficiency', title='Normalized Efficiency Scores at Percentiles')
    ax.barh(percentiles + 0.02, wiring[:, 1], 0.04, color='#81e2fc', label='wiring')
    ax.barh(percentiles - 0.02, routing[:, 1], 0.04, color='#5897a8', label='routing')
    ax.barh(percentiles + 0.02, -wiring[:, 0], 0.04, color='#03caff')
    ax.barh(percentiles - 0.02, -routing[:, 0], 0.04, color='#1388a8')
    ax.scatter((wiring[:, 1] - wiring[:, 0]) / 2, percentiles + 0.02, color='red', marker='x', zorder=5)
    ax.scatter((routing[:, 1] - routing[:, 0]) / 2, percentiles - 0.02, color='red', marker='x', zorder=5)
    ax.text(-0.5, 0.95, 'core', ha='center',)
    ax.text(0.5, 0.95, 'periphery', ha='center')
    ax.vlines(0, 0.0, 1, color='black')
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

    percentiles_bar_chart(dataset, centrality, method, ax, value_lines=True)
    plt.setp(ax, yticks=np.linspace(0, 1, 11), ylabel='% of total edges',
             title='$\\frac{num\ edges\ at\ percentile}{num\ edges}$ at different percentiles')
    ax.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 2, 'red')


def core_periphery_cost_at_percentiles(dataset, centrality, ax):
    def method(dataset, core):
        delta = np.asarray(np.meshgrid(core, core)).max(0)
        cost = np.multiply(dataset.adjacency, delta).sum() / dataset.adjacency.sum()
        return cost

    percentiles_bar_chart(dataset, centrality, method, ax, error_lines=True)
    plt.setp(ax, ylabel='core-periphery efficiency', title=f'core-periphery efficiency in percentiles & in random graphs')


def plot_percentile_modularity(dataset, centrality):
    charts = [
        (degree_utilization_in_percentiles, ),
        (number_of_edges_at_percentiles, ),
        (modularity_efficiency_at_percentile, ),
        (travel_costs_at_percentiles, ),
        (relative_convex_volume_at_percentiles, ),
        (core_periphery_cost_at_percentiles, ),
    ]
    for left, right in list(zip(range(0, len(charts), 2), list(range(1, len(charts), 2)) + [None])):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        left_method, left_args = charts[left][0], charts[left][1:] if len(charts[left]) > 1 else []
        left_method(dataset, centrality, *left_args, ax1)
        if right:
            right_method, right_args = charts[right][0], charts[right][1:] if len(charts[right]) > 1 else []
            right_method(dataset, centrality, *right_args, ax2)
        st.pyplot(fig)


def brain_centrality():
    brain_name, node_resistance, angle = _get_parameters()
    if brain_name == 'Random Model':
        brain = random_model()
    else:
        brain = load_brain(load_brain)

    centrality = resistance_centrality(brain, node_resistance)
    edge_centrality = edge_resistance_centrality(brain, node_resistance)
    plot_3d_brains(brain, centrality, angle)
    st.pyplot(plot_hist(np.array(brain.distances).flatten(), 20, 'distances', 'distance', 'freq'))
    st.pyplot(plot_hist((np.array(brain.adjacency) > 0).sum(0), 20, 'degree dist', 'degree', 'freq'))
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
