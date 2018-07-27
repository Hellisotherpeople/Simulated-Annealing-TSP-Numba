from functools import reduce
from random import randint, random, sample
import math



from scipy.spatial.distance import pdist
import numpy as np


cities = np.random.RandomState(22).rand(3000,2)
city_names = ("jonestown", "beaker_city", "Intel")
#cities = np.array([[17, 19], [2, 2], [8,8], [12, 14], [1, 1]])
print(cities)

def compute_path_travel_distance(cities):
    distance = 0
    for i in range(0, len(cities) - 1):
        distance += np.linalg.norm(cities[i]-cities[i+1])
    return distance
    #return reduce(lambda x, y: np.linalg.norm(x-y), cities)

print(compute_path_travel_distance(cities))

def reverse_random_sublist(lst):
     #FIX THIS 
    new_list = lst.copy()
    cities_len = len(cities)-1
    start = randint(0, cities_len)
    end = randint(start, cities_len)
    new_list[start:end+1] = new_list[start:end+1][::-1]
    return new_list
#print("--------------------")
#print(reverse_random_sublist(cities))

def random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return list(sample(pool, r))

def acceptance_probability(old_cost, new_cost, temperature):
    try:
        res = math.exp((old_cost - new_cost) / temperature) 
    except OverflowError:
        res = float('inf')
    return res
def simulated_annealing(cities):
    old_cost = compute_path_travel_distance(cities)
    temperature = 1.0
    min_temperature = 0.0001
    alpha = 0.90
    best_solution = None
    solution = cities 
    while temperature > min_temperature:
        iteration = 1 
        while iteration <= 500:
            #canidate = random_permutation(solution, r = len(cities)) #Not NEARLY as good as the other one
            canidate = reverse_random_sublist(solution)
            new_cost = compute_path_travel_distance(canidate)
            if new_cost < old_cost:
                best_solution = canidate
            if iteration % 50 == 0:
                a_sol = compute_path_travel_distance(best_solution)
                f_string = f"Iteration #: {iteration}, temperature: {temperature}, solution:  {old_cost}"
                print(f_string)
            ap = acceptance_probability(old_cost, new_cost, temperature)
            if ap > random():
                solution = canidate
                old_cost = new_cost
            iteration += 1
        temperature = temperature * alpha
    return solution, compute_path_travel_distance(solution), compute_path_travel_distance(best_solution)




print(simulated_annealing(cities))



