import math
import unittest


class MetricResult:
    def __init__(self, value, optimal_value=None, worst_value=None, mean_value=None):
        self.value = value
        self.optimal_value = optimal_value
        self.worst_value = worst_value
        self.mean_value = mean_value
        self._fill_values()

    def _fill_values(self):
        if self.worst_value and not self.mean_value:
            self.mean_value = (self.worst_value - self.optimal_value) / 2
        if self.mean_value and not self.worst_value:
            self.worst_value = self.mean_value + self.mean_value - self.optimal_value

    @property
    def normalized_value(self):
        if math.isinf(self.value):
            return float('-inf')
        if self.optimal_value is not None and self.worst_value is not None:
            return (self.worst_value - self.value) / (self.worst_value - self.optimal_value)
        return self.value

    def to_dict(self):
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "normalized_factor": self.optimal_value
        }


class MetricResultTests(unittest.TestCase):
    def test_no_normalization(self):
        value = 50
        mr = MetricResult(value)
        self.assertEquals(value, mr.normalized_value)

    def test_infinity(self):
        value = float('inf')
        mr = MetricResult(value)
        self.assertEquals(float('-inf'), mr.normalized_value)

    def test_optimal_and_worst(self):
        value = 50
        optimal = 0
        worst = 100
        mr = MetricResult(value, optimal, worst)
        self.assertEquals(0.5, mr.normalized_value)
        self.assertEquals(50, mr.mean_value)

    def test_optimal_and_mean(self):
        value = 50
        optimal = 0
        mean = 50
        mr = MetricResult(value, optimal, mean_value=mean)
        self.assertEquals(0.5, mr.normalized_value)
        self.assertEquals(100, mr.worst_value)
