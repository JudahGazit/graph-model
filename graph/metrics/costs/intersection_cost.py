import itertools

import numpy as np
import pandas as pd
from bresenham import bresenham

from graph.metrics.Metric import Metric, MetricBoundaries
from graph.metrics.costs.icost import ICost


class IntersectionCost(ICost):
    def __edges_intersections_with_blocks(self, grid_scale):
        node_positions = [grid_scale * np.array(self.graph_dataset.positions(i)) for i in range(self.graph_dataset.number_of_nodes)]
        edges = pd.DataFrame([[row] for row in self.graph_dataset.graph.get_edgelist()], columns=['edge'])
        edges['block'] = edges['edge'].apply(
            lambda row: tuple(bresenham(*node_positions[row[0]], *node_positions[row[1]])), )
        edges = edges.explode('block').groupby('block', as_index=False).agg(set)
        return edges

    def cost(self, grid_scale=2):
        if self.graph_dataset.positions is not None:
            blocks = self.__edges_intersections_with_blocks(grid_scale)
            blocks = blocks[blocks['block'].apply(lambda point: point[0] % grid_scale != 0 or point[1] % grid_scale != 0)]
            blocks = blocks[blocks['edge'].str.len() > 1]
            blocks['edge'] = blocks['edge'].apply(
                lambda intersection: [(*u, *v) for u, v in itertools.combinations(intersection, 2) if
                                      len({*u, *v}) == 4])
            edges = blocks.explode('edge').dropna()
            intersections = edges['edge'].drop_duplicates().size
            return Metric(intersections, self.boundaries)
        return Metric(-1)

    @property
    def boundaries(self) -> MetricBoundaries:
        return MetricBoundaries(100)
