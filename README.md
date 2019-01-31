# tsp-heuristics
Heuristic Algorithms for solving the Traveling Salesman Problem, developed for the 2018W - Computational Optimization course.

## Algorithms

### Local Search: ls_2opt.py

Multistart local search algorithm using a 2-opt neighbourhood structure.

Implements two initialization methods:
* Random Inititalization
* Nearest Neighbour Heuristic

Implements two neighbour selection strategies:
* Best Improvement
* First Improvement

### Genetic Algorithm: ga.py

Genetic local search algorithm mostly based on the papers [1, 2]. It reuses the local search
implementation from ls_2opt.py for the initial population generation and as the mutation operator.

Implements two different crossover strategies:
* Order Crossover OX as used in [2]
* Maximum Preservation Crossover [3]

[1] Ulder et al. (1991). Genetic local search algorithms for the 
traveling salesman problem.
    
[2] Braun H. (1991). On solving travelling salesman problems by
genetic algorithms.
		
[3] Muehlenbein, H. (1991). Evolution in Time and Space - The 
Parallel Genetic Algorithm

## TSP-Instances

Implementations have been tested on and nativly support the berlin52, gr120, pcb442 and fl1400 instances from TSPLIB
