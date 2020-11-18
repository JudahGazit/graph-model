import math


class MetricResult:
    def __init__(self, value, normalization_factor=1, normalized_value=None):
        self.value = value
        self.normalization_factor = normalization_factor if normalization_factor is not None else self.value * normalized_value
        self.normalized_value = normalized_value if normalized_value is not None else self.__calculate_normalized()

    def __calculate_normalized(self):
        if math.isinf(self.value):
            return float('-inf')
        return self.normalization_factor / self.value

    def to_dict(self):
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "normalized_factor": self.normalization_factor
        }
