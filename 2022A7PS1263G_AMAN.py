import sys
import random
import statistics as st
import json
import numpy as np
import math
from SetCoveringProblemCreator import SetCoveringProblemCreator
from matplotlib import pyplot as plt
import time 

class GeneticAlgorithm:
    def __init__(self, subsets, population_size=50, generations=50, alpha=1, beta=1):
        self.subsets = subsets
        self.universe = set(range(1, 101))
        self.alpha = alpha
        self.beta = beta
        self.population_size = population_size
        self.generations = generations
        self.population = self._initialize_population()

    def _initialize_population(self):
        # Randomly initialize the population with binary chromosomes
        return [np.random.randint(2, size=len(self.subsets)) for _ in range(self.population_size)]

    def _fitness(self, chromosome):
        # Decode the chromosome into a list of chosen subsets
        chosen_subsets = [self.subsets[i] for i, bit in enumerate(chromosome) if bit == 1]
        covered_elements = set().union(*chosen_subsets)
        
        # Fitness is determined by two factors:
        # 1. Coverage of the universe (penalty for uncovered elements)
        # 2. Minimizing the number of subsets chosen
        coverage_score = len(self.universe - covered_elements)
        subset_count = sum(chromosome)

        fitness_score = self.alpha * coverage_score * math.log(1 + coverage_score) + self.beta * subset_count * math.log(1 + subset_count)
        # Fitness function: Minimize uncovered elements and subset count
        return fitness_score, coverage_score

        fitness_score = coverage_score + subset_count
        
        # Fitness function: Minimize uncovered elements and subset count
        return fitness_score, coverage_score

    def _selection(self):
        # Select two parents using tournament selection
        tournament_size = 3
        selected = random.sample(self.population, tournament_size)
        selected.sort(key=self._fitness)
        return selected[0], selected[1]

    def _crossover(self, parent1, parent2):
        # Perform single-point crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def _mutation(self, chromosome, mutation_rate=0.001):
        # Mutate the chromosome with a small probability
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip the bit
        return chromosome

    def _evolve(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self._selection()
            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutation(child1)
            child2 = self._mutation(child2)
            new_population.extend([child1, child2])
        self.population = new_population

    def run(self):
        best_chromosome = None
        best_coverage_score = None
        best_fitness = float('inf')
        fitness_array = []
        mean_array = []
        stdev_array = []
        
        for generation in range(self.generations):
            self.population.sort(key=self._fitness)
            current_best = self.population[0]
            current_best_fitness, current_best_coverage_score = self._fitness(current_best)
            fitness_array.append(int(current_best_fitness))
            mean = st.mean(fitness_array)
            try:
                std_dev = st.stdev(fitness_array)
            except:
                std_dev = 0
            
            mean_array.append(mean)
            stdev_array.append(std_dev)
            
            if current_best_fitness < best_fitness:
                best_coverage_score = current_best_coverage_score
                best_fitness = current_best_fitness
                best_chromosome = current_best
            
            #print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
            self._evolve()
        
        return best_chromosome, best_fitness, best_coverage_score, mean_array, stdev_array

def main():
    scp = SetCoveringProblemCreator()
    
    # Read SCP instance from JSON file
    fileName = "scp_test_1.json"
    listOfSubsets = scp.ReadSetsFromJson(fileName)
    if listOfSubsets is None:
        return
    
    # Track start time
    start_time = time.time()
    
    # Run Genetic Algorithm
    ga = GeneticAlgorithm(listOfSubsets, population_size=150, generations=200, alpha=1, beta=1)
    best_solution, best_fitness, best_coverage_score, mean_array, stdev_array = ga.run()

    # Calculate time taken
    time_taken = time.time() - start_time

    # Plotting the results
    generations = list(range(1, len(mean_array) + 1))

    plt.figure(figsize=(12, 6))

    # Plot mean fitness values
    plt.plot(generations, mean_array, label='Mean Fitness', color='blue')

    # Plot standard deviation of fitness values
    plt.plot(generations, stdev_array, label='Standard Deviation', color='red')

    # Adding titles and labels
    plt.title('Fitness Metrics over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig("50plot1")

    # Show plot
    # plt.show()

    # Prepare the output format
    roll_number = "2022A7PS1263G"  # Replace this with your actual roll number
    total_subsets = len(listOfSubsets)
    selected_subsets = [f"{i}:{bit}" for i, bit in enumerate(best_solution)]
    fitness_value = int(best_fitness)
    min_subsets = sum(best_solution)
    
    # Output the results formatted like the screenshot
    print(f"Roll no : {roll_number}")
    print(f"Number of subsets in {fileName} file : {total_subsets}")
    print("Solution")
    print(" ".join(selected_subsets))
    print(f"Fitness value of best state : {fitness_value}")
    print(f"Minimum number of subsets that can cover the Universe-set : {min_subsets}")
    print(f"Time taken : {time_taken:.2f} seconds")

if __name__ == '__main__':
    main()