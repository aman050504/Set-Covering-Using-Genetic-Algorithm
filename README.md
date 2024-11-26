# Set Covering Problem (SCP) - Genetic Algorithm

This project implements a **Genetic Algorithm (GA)** to solve the **Set Covering Problem (SCP)**, where the goal is to find the minimum number of subsets needed to cover a universe set. The algorithm evolves through generations to identify the best possible solution based on a fitness function.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results and Evaluation](#results-and-evaluation)
- [License](#license)

## Project Overview
The **Set Covering Problem (SCP)** is a classic combinatorial optimization problem where the objective is to select the smallest number of subsets from a given collection of subsets that cover all elements in a universal set. This project applies a **Genetic Algorithm (GA)** to solve the SCP, optimizing the selection of subsets based on a custom fitness function.

The algorithm evolves over generations, adjusting the population through selection, crossover, and mutation operations to improve the solution. The goal is to minimize the number of subsets while ensuring full coverage of the universe.

## Requirements
To run the project, you'll need the following:
- Python 3.8 or later
- NumPy
- Matplotlib (for visualizations)
- JSON (for reading SCP problem data)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/set-covering-genetic-algorithm.git
   cd set-covering-genetic-algorithm
2. Install the required Python packages:

pip install -r requirements.txt

3. Ensure you have the correct versions of Python and the libraries mentioned.

## Usage

1. To run the genetic algorithm and solve the SCP, execute:
python solve_scp.py

2.The program will read an input file (scp_test.json) containing the SCP instance and output the solution with the selected subsets.

3.To visualize the performance over generations, run:
python plot_performance.py

4.This will generate graphs showing the progress of the genetic algorithm and its convergence to the optimal solution.


## How it Works
Genetic Algorithm (GA): The GA starts with a random population of solutions and evolves over multiple generations. Each solution represents a subset of subsets from the collection, aiming to cover all elements in the universe.

Fitness Function: The fitness function evaluates each solution based on two factors:

Coverage Score: How well the selected subsets cover the universe.

Subset Count: Penalizes solutions that use more subsets, encouraging the selection of fewer subsets.
Selection, Crossover, and Mutation:

Selection: Individuals (solutions) with better fitness values have a higher chance of being selected for reproduction.

Crossover: Two parent solutions are combined to create offspring, inheriting characteristics from both parents.

Mutation: A random change is introduced in the offspring to maintain diversity in the population.

Termination Criteria: The algorithm stops after a predefined number of generations or when the solution converges (i.e., the fitness value does not change significantly over several generations).

## Result and Evaluation

Performance: 
The genetic algorithm was evaluated on SCP instances with different numbers of subsets (50, 150, 250, 350). The algorithm's performance was analyzed by plotting the mean and standard deviation of the best fitness value over 50 generations.

Key Findings:
The genetic algorithm effectively minimized the number of subsets while ensuring full coverage of the universe.
The performance improved with larger population sizes and more generations.
The algorithm demonstrated steady convergence, although the solution quality varied depending on the SCP instance size.

Graphical Evaluation: 
The plots show how the fitness function evolves over generations, providing insight into the convergence behavior of the algorithm.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to replace the repository URL `https://github.com/yourusername/set-covering-genetic-algorithm.git` with your actual repository URL. Copy this content directly into your repository’s `README.md` file.




