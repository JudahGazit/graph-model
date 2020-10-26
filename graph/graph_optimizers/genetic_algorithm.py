import random

from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase


class GeneticAlgorithm(GraphOptimizerBase):
    def __score_and_sort(self, population):
        scores = [self.graph_cost.cost(self.graph_cost.triangular_to_mat(edges)) for edges in population]
        return sorted(zip(scores, population))

    def __selection(self, population, n_parents):
        return population[:n_parents]

    def __crossover(self, population):
        new_population = list(population)
        for index, edges in enumerate(population):
            first_indices = [i for i, v in enumerate(edges) if v == 1]
            second_indices = [i for i, v in enumerate(population[(index + 1) % len(population)]) if v == 1]
            new_generation = random.sample(list(set(first_indices + second_indices)), self.num_edges)
            new_generation = [1 if i in new_generation else 0 for i in range(self._total_possible_edges)]
            new_population.append(new_generation)
        return new_population

    def __mutation(self, population, mutation_rate):
        population_nextgen = []
        for edges in population:
            for j in range(len(edges)):
                if random.random() < mutation_rate:
                    edges[j] = 1 - edges[j]
            edges = random.sample([i for i, v in enumerate(edges) if v == 1], self.num_edges)
            edges = [1 if i in edges else 0 for i in range(self._total_possible_edges)]
            population_nextgen.append(edges)
        return population_nextgen

    def _optimal_matrix(self, n_gen=200, n_parents=10, mutation_rate=0.05):
        population = [self._randomize_edges() for _ in range(n_parents)]
        best_mat = []
        best_score = []
        for i in range(n_gen):
            scores, pop_after_fit = zip(*self.__score_and_sort(population))
            best_mat.append(pop_after_fit[0].copy())
            best_score.append(scores[0])
            pop_after_sel = self.__selection(pop_after_fit, n_parents)
            pop_after_cross = self.__crossover(pop_after_sel)
            population = self.__mutation(pop_after_cross, mutation_rate)
        best_score, best_mat = min(zip(best_score, best_mat))
        best_mat = self.graph_cost.triangular_to_mat(best_mat)
        return best_mat