from graph.metrics.Metric import MetricBoundaries
from graph.metrics.costs.icost import ICost


class ResistanceCost(ICost):
    def cost(self):
        pass

    @property
    def boundaries(self) -> MetricBoundaries:
        pass