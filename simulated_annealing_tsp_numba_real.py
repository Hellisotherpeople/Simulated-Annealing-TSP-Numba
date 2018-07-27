from functools import reduce
from random import randint, random, sample
import math


from scipy.spatial.distance import pdist
import numpy as np

from numba import jit, prange, njit

######### "Oops, I accidently achieved state of the art results" #########
######### a fast Traveling Salesman Problem solver in Numpy and Numba. #########





# 3000 cities, with a specific seed for reproducability
cities = np.random.RandomState(22).rand(3000, 2)
#cities = np.array([[17, 19], [2, 2], [8,8], [12, 14], [1, 1]])
# print(cities)


@njit(fastmath=True, parallel=True)
def compute_path_travel_distance(cities):
    distance = 0
    for i in prange(0, len(cities) - 1):  # "parallel programming is hard"
        distance += np.linalg.norm(cities[i]-cities[i+1])  # euclidean distance
    return distance
    # return reduce(lambda x, y: np.linalg.norm(x-y), cities)

# print(compute_path_travel_distance(cities))


@njit(fastmath=True)
def reverse_random_sublist(lst):

    # I read online that this was much better than a random permutation for getting convergence
    new_list = lst.copy()
    cities_len = len(cities)-1
    start = randint(0, cities_len)
    end = randint(start, cities_len)
    new_list[start:end+1] = new_list[start:end+1][::-1]
    return new_list


@njit
def random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return list(sample(pool, r))


@njit(fastmath=True)
def acceptance_probability(old_cost, new_cost, temperature):
    res = math.exp((old_cost - new_cost) / temperature)
    return res


@njit(fastmath=True)
def simulated_annealing(cities):
    old_cost = compute_path_travel_distance(cities)
    temperature = 1.0
    min_temperature = 0.00001
    alpha = 0.95
    #best_solution = None
    solution = cities
    while temperature > min_temperature:
        for iteration in range(1, 500):
            # canidate = random_permutation(solution, r = len(cities)) #Not NEARLY as good as the other one
            canidate = reverse_random_sublist(solution)
            new_cost = compute_path_travel_distance(canidate)
            # if new_cost < old_cost:
            #    best_solution = canidate
            if iteration % 50 == 0:
                print(iteration, temperature, old_cost)
                # f_string = f"Iteration #: {iteration}, temperature: {temperature}, solution:  {old_cost}" #numba doesn't like F strings
                # print(f_string)
            ap = acceptance_probability(old_cost, new_cost, temperature)
            if ap > random():
                solution = canidate
                old_cost = new_cost
        temperature = temperature * alpha
    return solution, compute_path_travel_distance(solution)


print(simulated_annealing(cities))
