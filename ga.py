########################################################################
# Author: Lukas Gosch
# Date: 18.1.2019
# Usage: python ga.py -h
# Description: 
#	Implements a genetic local search algorithm for the TSP problem
#	mostly based on the papers [1, 2]. It reuses the local search
#	implementation from ls_2opt.py for the initial population 
#	generation and as the mutation operator.
#	
#	Implements two different crossover strategies:
#	- Order Crossover OX as used in [2]
#	- Maximum Preservation Crossover [3]
#
#	Unterstands the berlin52, gr120, pcb442 and fl1400 sample instances
#	from TSPLIB.
#
#	[1] Ulder et al. (1991). Genetic local search algorithms for the 
#		traveling salesman problem.
#	[2] Braun H. (1991). On solving travelling salesman problems by
#		genetic algorithms.
#	[3] Muehlenbein, H. (1991). Evolution in Time and Space - The 
#		Parallel Genetic Algorithm
########################################################################


import sys
import argparse
import time
import random
import numpy as np
from ls_2opt import loadTSP
from ls_2opt import localSearch
from ls_2opt import localOptimize

def initPopulation(popsize, n, dm, maxiter):
	""" Initialize population for genetic algorithm. 
		
		Create popsize initial solutions by choosing popsize
		different starting cities and apply local search on them.
		The local search uses the nearest neighbourhood heuristic
		for its initial solution generation and then does a best
		improvement neighbourhood selection in a 2-opt neighbourhood.

		Timecomplexity: O(popsize * maxiter * n^2) or
						O(popsize * n^3) if n > maxiter
	"""
	# List of already used initial solutions for greedy heuristic
	start_l = []
	# Population of solutions stored as list of individuals
	solutions = []
	fitness = []
	# Perform multistart local search to initialize population
	i = 0
	for i in range(1,popsize+1):
		np.random.seed(i)
		# Generate an individual through local search in O(maxiter * n^3).
		s,cost,init_cost,t = localSearch(n, dm,
										init_strategy='greedy',
										n_strategy='best_improve',
										maxiter=maxiter,
										start_l=start_l)
		solutions.append(s)
		fitness.append(cost)
		if t == maxiter:
			print("Warning: maxiter reached in initializations local search.")
	# Return population as tuple of fitness and solutions
	return (np.array(solutions), np.array(fitness))

def selectParents(population, strategy='tournament', k=2):
	""" Returns two parents based on a given selection strategy. 
		
		Binary Tournament (k=2) is recommended based on computational
		efficiency (rank based roulette wheel needs sorting) and it's
		mathematical equivalence with linear normalization as shown
		by Julstrom in "It's All the Same to Me: Revisiting Rank-Based
		Probabilities and Tournaments" doi:10.1109/cec.1999.782661
	"""
	# k-tournament selection, O(n) (as k <= n)
	if strategy == 'tournament':
		I = [i for i in range(0, population[1].size)]
		parents = []
		for parent in [1, 2]:
			pool = random.sample(I,k) # O(k) NOTE: np.random.choice has an inefficient implementation for k << n, probably would need O(kn)
			# Choose individual with maximal fitness as parent
			rel_winner_i = np.argmin(population[1][pool]) # O(k)
			winner_i = pool[rel_winner_i]
			parents.append(population[0][winner_i])
			# Remove winner as not to choose him again, O(n)
			I.remove(winner_i)
		return parents[0], parents[1]

	sys.exit("Selection Mechanism for Parents not supported.")

def addEdge(child, parent, dm):
	""" If possible, add edge from parent to child.
		Time Complexity: O(n)
		Return: True if sucessfull
	"""
	cities = dm.shape[0]
	len_o = len(child)
	l_c = child[len_o - 1]
	# Find Index of last_city in parent
	r_i = parent.index(l_c)
	# Get previous and next city in parent
	if r_i > 0:
		p_c = parent[r_i - 1]
	else:
		p_c = parent[cities - 1]
	if r_i < cities - 1:
		n_c = parent[r_i + 1]
	else:
		n_c = parent[0]
	# If possible, add edge from parent to child, O(n)
	p_flag = True
	n_flag = True
	if p_c in child:
		p_flag = False
	if n_c in child:
		n_flag = False
	if p_flag and n_flag:
		if dm[l_c,p_c] < dm[l_c,n_c]:
			child.append(p_c)
		else: 
			child.append(n_c)
	elif p_flag:
		child.append(p_c)
	elif n_flag:
		child.append(n_c)
	else:
		return False
	return True

def crossover(p1, p2, strategy='MPX', dm=None):
	""" Cross the two parents p1 & p2 to generate an offspring.

		Implements order crossover OX and maximum preservation
		crossover MPX operator.

		Time complexity:
		- OX: O(n^2)
		- MPX: O(n^3)

		Details:
		- OX implementation follows definition by 
		  Braun in "On Solving Travelling Salesman Problems by
		  Genetic Algorithms", 1990.
		  OX is the recommended operator for the chosen
		  path representation of the cities by Potvin in
		  "Genetic algorithms for the traveling salesman problem",
		  Annals of Operations Research 63(1996)339-370
		- MPX implementation follows definition by
	      Muehlenbein in "Evolution in Time and Space - The Parallel
	      Genetic Algorithm", 1991. In general produces more
	      genetically divers offspring and is the recommended
	      crossover operator.
	""" 

	# Time Complexity: O(n^2)
	# Proof: O(kn) most time complex operation in loop,
	#		 loop is repeated at most n/k + 1 times
	if strategy == 'OX':
		p1 = p1.tolist()
		p2 = p2.tolist()
		# Save last city in p1 for later appending
		last_city = p1.pop()
		# Delete last city in p2 as p1's starting city is used
		del p2[len(p2)-1]
		k = int(1/3 * len(p1)) 
		offspring = []
		# Repetitions: n / k upper rounded
		while p1:
			p11_len = min(len(p1),k)
			p11 = p1[:p11_len]
			offspring.extend(p11)
			p1 = p1[p11_len:]
			# O(kn) 
			p2 = [city for city in p2 if city not in p11]
			# Swap p1 and p2 (O(1), only name bindings)
			tmp = p1
			p1 = p2
			p2 = tmp
		offspring.append(last_city)
		return offspring
	# Time Complexity: O(n^3) (Worst Case)
	if strategy == 'MPX':
		donor = p1.tolist()
		receiver = p2.tolist()
		# Delete last city which is same as first
		cities = len(donor) - 1
		del donor[cities]
		del receiver[cities]
		i = np.random.randint(0,cities)
		k = np.random.randint(int(0.2 * cities), int(0.4 * cities))
		# Extract Crossover String
		offspring = []
		j = (i + k) % cities
		if j > i:
			offspring = donor[i:j+1]
		else:
			offspring = donor[i:] + donor[:j+1]
		# Add edges to offspring until a valid tour, Worst Case: O(n^3)
		while len(offspring) < cities:
			# Try to add edge from receiver to offspring
			if not addEdge(offspring, receiver, dm):
				# Try to add edge from donor to offspring
				if not addEdge(offspring, donor, dm):
					# Add city from receiver not in offspring which 
					# is next in list
					len_o = len(offspring)
					l_c = offspring[len_o - 1]
					r_i = receiver.index(l_c)
					c_i = (r_i + 2) % cities
					while receiver[c_i] in offspring:
						c_i += 1
						c_i = c_i % cities
					offspring.append(receiver[c_i])
		offspring.append(offspring[0])
		return offspring

	sys.exit("Choosen Crossover Strategy not supported.")

def main(argv):
	parser = argparse.ArgumentParser(description='Solve TSP with a genetic local search.')
	parser.add_argument('path', help='Path to .tsp file.')
	parser.add_argument('-ps', '--popsize', help='Size of the population.', type=int, default=10)
	parser.add_argument('-mg', '--maxgen', help='Maximum number of generations.', type=int, default=10000)
	parser.add_argument('-mi', '--maxiter', help='Maximum number of iterations in each local search.', type=int, default=1000)
	parser.add_argument('-sel', '--selection', help='Define selection strategy for parents.', choices=['tournament'], default='tournament')
	parser.add_argument('-co', '--crossover', help='Define crossover strategy. ', choices=['OX','MPX'], default='MPX')
	parser.add_argument('-st', '--stop', help='How many generation should produce no better offspring for the algorithm to stop.', type=int, default=-1)
	args = parser.parse_args(argv[1:])

	# If no or invalid stopping criterion is given, chose "population 
	# size" as maximum number of generations for no valid offspring
	if args.stop < 1:
		args.stop = args.popsize

	# Create distance matrix based on .tsp file
	dm = loadTSP(args.path)
	n = dm.shape[0]

	start_t = time.process_time()
	# Initialize Population in O(popsize * maxiter * n^2) or O(popsize * n^3) if n > maxiter
	population = initPopulation(args.popsize, n, dm, args.maxiter)
	end_t = time.process_time()
	diff_t_init = end_t - start_t
	print("Successfully generated ", args.popsize, " individuals in ", diff_t_init,"s.")
	print("Avg. time per individual: ", diff_t_init / args.popsize)

	start_t = time.process_time()
	best_fitness = min(population[1])
	best_gen = 0
	counter = 0
	worst_fitness = max(population[1])
	worst_index = np.argmax(population[1]) 
	print("Worst fitness: ", worst_fitness)
	# Evolve Population in O(gen * maxiter * n^2) or O(gen * n^3) if n > maxiter
	for gen in range(0,args.maxgen):
		# Check stopping criterion
		if counter >= args.stop:
			print("Genetic Local Search converged in generation ", gen, ".",
				  "Could not generate a better offspring for ", counter, 
				  " generations.")
			break
		counter += 1
		random.seed(gen)
		# Select Parents for Offspring generation, O(n)
		p1, p2 = selectParents(population, args.selection)
		# Create Offspring, O(n^2) (OX) or O(n^3) (MPX)
		child = crossover(p1, p2, args.crossover, dm)
		# Locally Optimize Offspring, O(maxiter * n^2)
		child, cost, init_cost, t = localOptimize(child,dm,'best_improve',args.maxiter)
		# Replace worst individual with child if it is fitter, O(n)
		if cost < worst_fitness:
			# Check if fitness value already in population
			if not np.isin([cost], population[1])[0]:
				population[0][worst_index] = child
				population[1][worst_index] = cost
				# Update worst individual
				worst_fitness = max(population[1])
				worst_index = np.argmax(population[1]) 
				# Statistics
				if cost < best_fitness:
					best_fitness = cost
					best_gen = gen
				print("Generation ", gen, " produced a fitter child with cost: ", cost)
				counter = 0
	best_fitness = min(population[1])
	end_t = time.process_time()
	diff_t = end_t - start_t
	print("Successfully evolved", gen," generations in ", diff_t,"s.")
	print("Avg. time per generation: ", diff_t / gen)
	print("Best solution cost: ", best_fitness, " evolved in generation: ", best_gen)
	print("Avg. solution quality: ", sum(population[1]) / args.popsize)
	print("Generated ", args.popsize, " individuals in ", diff_t_init,"s.")
	print("Avg. time per individual for initialization: ", diff_t_init / args.popsize)

if __name__ == '__main__':
	main(sys.argv)