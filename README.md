# Simulated-Annealing-TSP-Numba
Using a metaheaurisitic technique called Simulated Annealing to solve the Traveling Salesman Problem. 

The numba version is about 100x faster than the pure python version. Also, it's parallelized and will use as many CPU cores as you'll give it. It respects things like taskset and numactl for reducing the number of CPUs used. 
