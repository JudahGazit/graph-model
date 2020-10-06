class MetricResult:
    def __init__(self, value, normalization_factor=1):
        self.__value = value
        self.__normalization_factor = normalization_factor

    @property
    def value(self):
        return self.__value

    @property
    def normalized_value(self):
        return self.__value / self.__normalization_factor

    def to_dict(self):
        return {
            "value": self.value,
            "normalized_value": self.normalized_value
        }
