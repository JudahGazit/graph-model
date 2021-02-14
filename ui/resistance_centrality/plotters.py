import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.spatial
import scipy.stats
from matplotlib import animation

from graph.graph_dataset import GraphDataset
from graph.graph_formatter import GraphFormatter


def plot(graph_dataset: GraphDataset, node_colors=None, with_labels=True, mark_min=False, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(17, 14))
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
            ax.scatter(position[0], position[1], c='black', linewidths=100, zorder=10)
    else:
        node_colors = None
    nx.draw_networkx(graph_dataset.nx_graph,
                     pos=[graph_dataset.positions(i) for i in range(graph_dataset.number_of_nodes)],
                     nodelist=range(graph_dataset.number_of_nodes),
                     node_color=node_colors, cmap=cmap, with_labels=with_labels, node_size=100, ax=ax)
    return fig


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
    if color:
        colors, mappable, num_nodes = color_fiber_length(dataset, x, centrality, method=method)
        plt.colorbar(mappable)
        for i, n in enumerate(num_nodes):
            ax.text(i, 0, n, horizontalalignment='center', fontsize=8)
    plt.setp(ax, title='fiber length distribution' + f', colored by {method} centrality of nodes' if color else '',
             xticks=[], ylim=(0, 1))
    ax.bar(range(len(x)), chart['y'], width=1, color=colors)
    return fig


def plot_scatter(x, y, c, title, xlabel, ylabel, colorbar=False, cmap=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    plt.setp(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.scatter(x, y, c=c, cmap=cmap)
    if colorbar:
        mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                         norm=plt.Normalize(vmin=c.min(), vmax=c.max()))
        plt.colorbar(mappable)
    return fig


def plot_hist(x, bins, title, xlabel, ylabel, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    plt.setp(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.hist(x, bins=bins, density=True)
    return fig


def plot_brain_point_cloud(dataset, centrality, angle, fig=None, ax=None):
    if ax is None or fig is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    points = np.array([dataset.positions(i) for i in range(dataset.number_of_nodes)])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=centrality)
    ax.view_init(elev=10., azim=angle)
    return fig
