import logging

from graph.metrics.Metric import MetricBoundaries
from graph.metrics.costs.fuel_cost import FuelCost
from graph.metrics.costs.intersection_cost import IntersectionCost
from graph.metrics.costs.modularity_cost import ModularityCost
from graph.metrics.costs.routing_cost import RoutingCost
from graph.metrics.costs.wiring_cost import WiringCost

logger = logging.getLogger('metrics')


class CostBoundaries:
    def __init__(self, wiring=None, routing=None, fuel=None):
        self.wiring = wiring or MetricBoundaries()
        self.routing = routing or MetricBoundaries()
        self.fuel = fuel or MetricBoundaries()


class GraphMetrics:
    def __init__(self, graph_dataset, cost_boundaries=None):
        self.cost_boundaries = cost_boundaries or CostBoundaries()
        self._wiring = WiringCost(graph_dataset, self.cost_boundaries.wiring)
        self._routing = RoutingCost(graph_dataset, self.cost_boundaries.routing)
        self._fuel = FuelCost(graph_dataset, self.cost_boundaries.fuel)
        self._intersection = IntersectionCost(graph_dataset, None)
        self._modularity = ModularityCost(graph_dataset, None)
        self.cost_boundaries = self.__update_cost_boundaries(cost_boundaries)

    def __update_cost_boundaries(self, cost_boundaries):
        cost_boundaries = cost_boundaries if cost_boundaries is not None else CostBoundaries()
        cost_boundaries.wiring = self._wiring.boundaries
        cost_boundaries.routing = self._routing.boundaries
        cost_boundaries.fuel = self._fuel.boundaries
        return cost_boundaries

    def all_path_lengths(self, weight: bool):
        return self._routing.all_path_lengths(weight)

    def wiring_cost(self):
        logger.debug('start wiring cost')
        result = self._wiring.cost()
        logger.debug('end wiring cost')
        return result

    def routing_cost(self):
        logger.debug('start routing cost')
        result = self._routing.cost()
        logger.debug('end routing cost')
        return result

    def fuel_cost(self):
        logger.debug('start fuel cost')
        result = self._fuel.cost()
        logger.debug('end fuel cost')
        return result

    def collision_cost(self, grid_scale=2):
        logger.debug('start intersection cost')
        result = self._intersection.cost()
        logger.debug('end intersection cost')
        return result

    def modularity_cost(self):
        return self._modularity.cost()
