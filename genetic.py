import numpy as np
import random
import TSPClasses
import time

'''Hyperparameters'''
max_unchanged_generations = 50
population_size = 50
mutation_rate = 20 # percent chance each gene will mutate
elite_size = 10
'''Hyperparameters'''
print_info = True


def solve(initial_solution, max_time, time_started):
    population = Population(initial_solution)
    initial_cost = population.fittest.cost

    changes = 0
    # Keep going until enough generations have passed with no changes OR if we've taken too long
    while changes <= max_unchanged_generations and time.time() - time_started < max_time:
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


'''Select the a subsection of the first parent x. Add it to the child. Add every city that is not already in the child
   to the child in the order that they're found in y '''
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
    def __init__(self, initial_solutions):
        self.gen_count = 0
        self.generation = []
        for s in initial_solutions:
            self.generation.append(Individual(s))
        self.generation.sort(key=lambda i: i.cost)
        self.fittest = self.generation[0]

    def evolve(self):
        self.mutate_population() # Get any useful mutations
        self.breed_population() # Get any useful recombinations
        self.generation.sort(key=lambda i: i.cost) # Sort the list with the most fit first
        self.fittest = self.generation[0] # Get the best guy
        self.gen_count += 1

        if print_info:
            print("Generation " + str(self.gen_count))
            print(" Fittest: " + str(self.fittest.cost))
            pass

    # Run crossover on each member of the population. There's a 50% chance to breed with a member of the elite, and a
    # 50% chance to breed with a random population member. If the offspring is less fit, discard it
    def breed_population(self):
        for i in range(0, population_size - 1):
            if random.randint(1, 2) == 1:
                breed_index = random.randint(0, population_size-1)
            else:
                breed_index = random.randint(0, elite_size)
            other = self.generation[breed_index]
            offspring = crossover(self.generation[i], other)
            if offspring.get_fitness() > self.generation[i].get_fitness():
                self.generation[i] = offspring
                print("mated")

    # Go through each member of the non-elite and mutate their route. If the change is not favorable, discard.
    def mutate_population(self):
        for i in range(elite_size, population_size - 1):
            offspring = self.generation[i].mutate()
            if offspring.get_fitness() > self.generation[i].get_fitness():
                self.generation[i] = offspring
                print("mutated")


class Individual:
    def __init__(self, cities):
        self.cities = cities
        self.solution = TSPClasses.TSPSolution(cities)
        self.cost = self.solution.cost

    # Go through each city in the route and swap it with a random other city. This has a certain percent chance to
    # occur for each city
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
