import numpy as np
import random
import TSPClasses
import time

'''Hyperparameters'''
max_unchanged_generations = 50
population_size = 50
mutation_rate = 10 # percent chance each gene will mutate
elite_size = 10
'''Hyperparameters'''
print_info = True


def solve(initial_solution, max_time):
    population = Population(initial_solution)
    initial_cost = population.fittest.cost

    changes = 0
    start_time = time.time()
    while changes <= max_unchanged_generations and time.time() - start_time < max_time:
        prev_fitness = population.fittest.get_fitness()
        population.evolve()
        if population.fittest.get_fitness() == prev_fitness:
            changes += 1
        else:
            changes = 0

    if print_info:
        print("Initial cost: " + str(initial_cost))
        print("Final cost: " + str(population.fittest.cost))
    return population.fittest.solution


def crossover(x, y):
    route = []

    city1 = random.randint(0, len(x.cities)-1)
    city2 = random.randint(0, len(x.cities)-1)

    start = min(city1, city2)
    finish = max(city1, city2)

    for i in range(start, finish):
        route.append(y.cities[i])

    route_part2 = [item for item in x.cities if item not in route]

    route += route_part2
    child = Individual(route)

    if child.cost > x.cost:
        return x
    return child


class Population:
    def __init__(self, initial_solution):
        self.fittest = Individual(initial_solution)
        self.gen_count = 0
        self.generation = np.empty(population_size)
        self._init_population()

    def _init_population(self):
        new_generation = []
        pioneer = self.fittest
        for _ in range(0, population_size):
            offspring = pioneer.mutate()
            new_generation.append(offspring)
            if offspring.get_fitness() > self.fittest.get_fitness():
                self.fittest = offspring
        self.generation = new_generation

    def evolve(self):
        self.mutate_population()
        self.breed_population()
        self.generation.sort(key=lambda i: i.cost)
        self.fittest = self.generation[0]
        self.gen_count += 1

        if print_info:
            print("Generation " + str(self.gen_count))
            print(" Fittest: " + str(self.fittest.cost))

    def breed_population(self):
        for i in range(elite_size, population_size - 1):
            elite = self.generation[random.randint(0, elite_size - 1)]
            offspring = crossover(self.generation[i], elite)
            if offspring.get_fitness() > self.generation[population_size - i - 1].get_fitness():
                self.generation[population_size - i - 1] = offspring

    def mutate_population(self):
        for i in range(0, population_size - 1):
            offspring = self.generation[i].mutate()
            if offspring.get_fitness() > self.generation[i].get_fitness():
                self.generation[i] = offspring


class Individual:
    def __init__(self, cities):
        self.cities = cities
        self.solution = TSPClasses.TSPSolution(cities)
        self.cost = self.solution.cost

    def mutate(self):
        new_solution = np.array(self.cities)
        for swap in range(len(new_solution)):
            if random.randint(1, 100) <= mutation_rate:
                swap_with = random.randint(0, len(self.cities)-1)
                new_solution[swap], new_solution[swap_with] = new_solution[swap_with], new_solution[swap]

        offspring = Individual(new_solution)
        return offspring

    def get_fitness(self):
        return -self.cost
