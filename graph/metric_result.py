class MetricResult:
    def __init__(self, value, normalization_factor=1, normalized_value=None):
        self.value = value
        self.normalization_factor = normalization_factor or self.value * normalized_value
        self.normalized_value = normalized_value or self.normalization_factor / self.value

    def to_dict(self):
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "normalized_factor": self.normalization_factor
        }
