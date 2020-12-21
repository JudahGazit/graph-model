import math
import unittest
from collections import namedtuple
from dataclasses import dataclass


@dataclass
class MetricBoundaries:
    optimal_value: float = None
    worst_value: float = None
    mean_value: float = None


class Metric:
    def __init__(self, value, metric_boundaries=None):
        self.value = value
        self.metric_boundaries = metric_boundaries or MetricBoundaries()
        self._fill_values()

    def _fill_values(self):
        optimal_value, worst_value, mean_value = self.metric_boundaries.optimal_value, self.metric_boundaries.worst_value, self.metric_boundaries.mean_value

        if worst_value is not None and mean_value is None:
            mean_value = (worst_value - optimal_value) / 2
        if mean_value is not None and worst_value is None:
            worst_value = mean_value + mean_value - optimal_value

        self.metric_boundaries.optimal_value, self.metric_boundaries.worst_value, self.metric_boundaries.mean_value = optimal_value, worst_value, mean_value

    @property
    def normalized_value(self):
        optimal_value, worst_value = self.metric_boundaries.optimal_value, self.metric_boundaries.worst_value
        if math.isinf(self.value):
            return float('-inf')
        if optimal_value is not None and worst_value is not None:
            return (worst_value - self.value) / (worst_value - self.metric_boundaries.optimal_value)
        elif optimal_value is not None:
            return optimal_value / self.value
        return self.value

    def to_dict(self):
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "normalized_factor": self.metric_boundaries.optimal_value,
            "optimal_value": self.metric_boundaries.optimal_value,
            "worst_value": self.metric_boundaries.worst_value,
        }


class MetricResultTests(unittest.TestCase):
    def test_no_normalization(self):
        value = 50
        mr = Metric(value)
        self.assertEquals(value, mr.normalized_value)

    def test_infinity(self):
        value = float('inf')
        mr = Metric(value)
        self.assertEquals(float('-inf'), mr.normalized_value)

    def test_optimal_and_worst(self):
        value = 50
        optimal = 0
        worst = 100
        mr = Metric(value, MetricBoundaries(optimal, worst))
        self.assertEquals(0.5, mr.normalized_value)
        self.assertEquals(50, mr.metric_boundaries.mean_value)

    def test_optimal_and_mean(self):
        value = 50
        optimal = 0
        mean = 50
        mr = Metric(value, MetricBoundaries(optimal, mean_value=mean))
        self.assertEquals(0.5, mr.normalized_value)
        self.assertEquals(100, mr.metric_boundaries.worst_value)

    def test_only_optimal(self):
        value = 2.5476
        optimal = 1.7917
        mr = Metric(value, MetricBoundaries(optimal))
        self.assertAlmostEqual(0.7033, mr.normalized_value, 4)
