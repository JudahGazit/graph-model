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
            new_population.append([1 if i in new_generation else 0 for i in range(self._total_possible_edges)])
        return new_population

    def __mutation(self, population, mutation_rate):
        population_nextgen = []
        for edges in population:
            for j in range(len(edges)):
                if random.random() < mutation_rate:
                    random_u = random.randint(0, len(edges) - 1)
                    random_v = random.randint(0, len(edges) - 1)
                    edges[random_u], edges[random_v] = edges[random_v], edges[random_u]
            population_nextgen.append(edges)
        return population_nextgen

    def _optimal_matrix(self, n_gen=100, n_parents=10, mutation_rate=0.05):
        population = [self._randomize_edges() for _ in range(n_parents)]
        best_mat = []
        best_score = []
        for i in range(n_gen):
            scores, pop_after_fit = zip(*self.__score_and_sort(population))
            pop_after_sel = self.__selection(pop_after_fit, n_parents)
            pop_after_cross = self.__crossover(pop_after_sel)
            population = self.__mutation(pop_after_cross, mutation_rate)
            best_mat.append(pop_after_fit[0])
            best_score.append(scores[0])
        best_score, best_mat = min(zip(best_score, best_mat))
        best_mat = self.graph_cost.triangular_to_mat(best_mat)
        return best_mat