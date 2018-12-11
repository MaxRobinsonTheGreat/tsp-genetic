#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import itertools
import heapq
import genetic


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        cities = set(self._scenario.getCities())
        bssf: TSPSolution = None
        count = 0

        start_time = time.time()
        for city in cities:
            if time.time() - start_time > time_allowance:
                break
            count += 1
            other_cities = cities.copy()
            other_cities.remove(city)
            route = [city]

            while other_cities:
                last_city = route[-1]
                closest = min(other_cities, key=lambda c: last_city.costTo(c))
                route.append(closest)
                other_cities.remove(closest)

                # check if we hit a dead end
                if last_city.costTo(closest) == math.inf:
                    break
            greedy_solution = TSPSolution(route)

            # if not a possible path, don't count it
            if greedy_solution.cost == math.inf:
                count -= 1

            bssf = min(bssf, greedy_solution, key=lambda s: s.cost) if bssf is not None else greedy_solution
        end_time = time.time()
        return {'cost': bssf.cost, 'time': end_time - start_time, 'count': count, 'soln': bssf,
                'max': None, 'total': None, 'pruned': None}

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = self.defaultRandomTour()
        bssf = results['cost']
        cities = self._scenario.getCities()

        pq = []  # priority queue
        root_node = node(cities[0], cities)
        init_root(root_node, cities)
        # print(root_node.matrix)
        root_node.update()
        heapq.heappush(pq, (root_node.get_priority(), root_node))

        time_started = time.time()
        final_node = None
        pruned = 0
        total_nodes = 0
        max_queue = 0
        bssf_updates = 0
        # print(root_node.matrix)
        # print(root_node.lb)
        while pq and time.time() - time_started < time_allowance:
            cost, n = heapq.heappop(pq)
            if n.lb > bssf:
                pruned += 1
                continue
            # print("PARENT - " + str(n.city._index))
            # print(n.matrix)
            for city in n.cities:
                total_nodes += 1
                child_node = node(city, n.cities)
                child_node.set_parent(n)
                child_node.update()
                # print("CHILD - " + str(child_node.city._index))
                # print(child_node.matrix)
                if child_node.lb < bssf:
                    if not child_node.cities:
                        # print("found better solution")
                        bssf_updates += 1
                        bssf = child_node.lb
                        final_node = child_node
                    else:
                        # print("added new node")
                        heapq.heappush(pq, (child_node.get_priority(), child_node))
                        if len(pq) > max_queue:
                            max_queue = len(pq)
        route = extract_path(final_node)
        if not route:
            solution = results['soln']
        else:
            solution = TSPSolution(route)
        results['cost'] = solution.cost
        results['time'] = time.time() - time_started
        results['count'] = bssf_updates
        results['soln'] = solution
        results['max'] = max_queue
        results['total'] = total_nodes
        results['pruned'] = pruned
        return results



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        solutions = []
        for _ in range(genetic.population_size):
            results = self.greedy(time_allowance)
            solutions.append(results['soln'].route)

        time_started = time.time()
        solution = genetic.solve(solutions, time_allowance)

        return {'cost': solution.cost, 'time': time.time() - time_started, 'count': None, 'soln': solution,
                'max': None, 'total': None, 'pruned': None}


def init_root(root, cities):
    matrix_size = len(root.cities)+1
    root.matrix = np.empty([matrix_size, matrix_size])
    for city in cities:
        for next_city in cities:
            if city is next_city:
                root.matrix[city._index][next_city._index] = np.inf
            cost = city.costTo(next_city)
            root.matrix[city._index][next_city._index] = cost
    root.lb = 0
    root.parent = None
    root.level = 0


def extract_path(n):
    cities = []
    if n is None:
        return cities
    while True:
        # print(n.city._index)
        cities.insert(0, n.city)
        if n.parent is None:
            return cities
        n = n.parent


class node:
    level_cost = 100

    def __init__(self, city, cities):
        self.city = city
        self.cities = list(cities)
        self.cities.remove(city)

    def set_parent(self, parent):
        self.matrix = np.array(parent.matrix)
        self.lb = parent.lb + self.matrix[parent.city._index][self.city._index]
        self.parent = parent
        self.level = parent.level + 1
        self.clean_matrix()
        self.update()

    def get_priority(self):
        return self.lb - node.level_cost * self.level

    def clean_matrix(self):
        if self.parent is None:
            return
        pi = self.parent.city._index
        i = self.city._index
        self.matrix[i][pi] = np.inf
        for r in range(len(self.matrix)):
            self.matrix[r][i] = np.inf
        for c in range(len(self.matrix[0])):
            self.matrix[pi][c] = np.inf

    def update(self):
        for r in range(len(self.matrix[0])):
            row = self.matrix[r]
            lowest = np.inf
            for n in row:
                if n < lowest:
                    lowest = n
            if lowest is not np.inf:
                self.matrix[r] = [x - lowest for x in row]
                self.lb += lowest

        for c in range(len(self.matrix[0])):
            lowest = np.inf
            for r in range(len(self.matrix)):
                n = self.matrix[r][c]
                if n < lowest:
                    lowest = n
            if lowest is not np.inf:
                for r in range(len(self.matrix)):
                    self.matrix[r][c] -= lowest
                self.lb += lowest

    def __lt__(self, other):
        return self.get_priority() < other.get_priority()
