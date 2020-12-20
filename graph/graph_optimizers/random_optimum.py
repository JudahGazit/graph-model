import math

from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase


def stirling_approx_ln_choose(n, k):
    res = k * math.log(n * math.e / k)
    res += -0.5 * math.log(2 * math.pi * k)
    res += - (k ** 2) / (2 * n)
    return res


class RandomOptimum(GraphOptimizerBase):
    def __randomize_matrix(self):
        edges_vec = self.randomize_edges()
        mat = self.graph_cost.triangular_to_mat(edges_vec)
        value = self.graph_cost.cost(mat)
        return mat, value

    def __number_of_iterations(self):
        iterations = 100 * stirling_approx_ln_choose(self._total_possible_edges,
                                                     self.num_edges)  # Stirling Approx. of ln(n choose k) for n >> k
        iterations = max([int(iterations), 10000])
        iterations = min([iterations, 15000])
        return iterations

    def _optimal_matrix(self):
        iterations = self.__number_of_iterations()
        min_value, min_arg = None, None
        for iteration in range(iterations):
            mat, value = self.__randomize_matrix()
            if min_value is None or value < min_value:
                min_value = value
                min_arg = mat
        return min_arg
