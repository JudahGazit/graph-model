import numpy as np
import logging
import random

from graph.graph_optimizers.graph_optimizer_base import GraphOptimizerBase


logger = logging.getLogger('genetic')


class GeneticAlgorithm(GraphOptimizerBase):
    def __init__(self, num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type, n_gen=590, n_parents=100, mutation_rate=0.002, tol=80):
        super().__init__(num_nodes, num_edges, wiring_factor, routing_factor, fuel_factor, method, cost_type)
        self.n_gen = n_gen
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.tol = tol

    def __score_and_sort(self, population):
        scores = [self.graph_cost.cost(self.graph_cost.triangular_to_mat(edges)) for edges in population]
        return sorted(zip(scores, population))

    def __selection(self, population, n_parents):
        return population[:n_parents]

    def __crossover(self, population):
        new_population = list(population)
        for index, edges in enumerate(population):
            first_indices = np.flatnonzero(edges).tolist()
            second_indices = np.flatnonzero(population[(index + 1) % len(population)]).tolist()
            new_generation_indices = set(random.sample(list(set(first_indices + second_indices)), self.num_edges))
            new_generation = np.zeros(self._total_possible_edges, np.int)
            for new_generation_index in new_generation_indices:
                new_generation[new_generation_index] = 1
            new_population.append(new_generation.tolist())
        return new_population

    def __mutation(self, population, mutation_rate):
        population_nextgen = []
        for edges in population:
            for j in range(int(len(edges) * mutation_rate)):
                random_u, random_v = random.randrange(len(edges)), random.randrange(len(edges))
                edges[random_u], edges[random_v] = edges[random_v], edges[random_u]
            population_nextgen.append(edges)
        return population_nextgen

    def __select_best(self, best_mat, best_score, best_score_index, current_iteration, current_best_matrix, current_best_score):
        current_best_score = round(current_best_score, 6)
        best_score_index = current_iteration if best_score is None or best_score > current_best_score else best_score_index
        best_score = current_best_score if best_score is None else min([current_best_score, best_score])
        best_mat = current_best_matrix[0].copy() if best_mat is None or current_best_score == best_score else best_mat
        return best_mat, best_score, best_score_index

    def _optimal_matrix(self):
        population = [self._randomize_edges() for _ in range(self.n_parents)]
        best_mat = None
        best_score = None
        best_score_index = None
        for i in range(self.n_gen):
            if (best_score_index is None or best_score_index > i - self.tol) and (best_score is None or -1 < best_score < 1):
                scores, pop_after_fit = zip(*self.__score_and_sort(population))
                best_mat, best_score, best_score_index = self.__select_best(best_mat, best_score, best_score_index, i, pop_after_fit, scores[0])
                logger.info('Genetic Optimizing - %s, best score = %f, best index = %d', i, best_score, i - best_score_index)
                pop_after_sel = self.__selection(pop_after_fit, self.n_parents)
                pop_after_cross = self.__crossover(pop_after_sel)
                population = self.__mutation(pop_after_cross, self.mutation_rate)
        best_mat = self.graph_cost.triangular_to_mat(best_mat)
        return best_mat