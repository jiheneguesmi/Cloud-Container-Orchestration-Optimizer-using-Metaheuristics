Cloud Container Orchestration Optimizer
Project Overview

A complete optimization framework for solving the Cloud Container Orchestration Problem (CCOP) using heuristics and metaheuristics.
This project applies operations research and advanced optimization techniques to modern cloud orchestration challenges.

Key Features

Multiple optimization approaches: Heuristic, Tabu Search, Artificial Immune System

Realistic cloud constraints: resource limits, anti-affinity rules, communication costs

Comprehensive analysis with comparisons and visualizations

Modular codebase allowing easy extension and customization

Project Structure
src/
├── models.py              # Data classes: Container, Server, ProblemInstance
├── heuristics.py          # Best-Fit with Communication Awareness algorithm
├── metaheuristics.py      # Tabu Search & Artificial Immune System
├── utils.py               # Analysis, visualization, utilities
├── main.py                # Main runner & method comparison
└── requirements.txt       # Dependencies

Performance Highlights
Method	Total Cost	Servers Used	Time (s)	Improvement
Heuristic	13,671.82	8 / 10	0.12	Baseline
Tabu Search	7,051.76	6 / 10	7.59	-48.4%
Immune System	9,172.56	8 / 10	5.00	-32.9%
Installation
1. Clone the repository
git clone <repository-url>
cd src

2. Install dependencies
pip install -r requirements.txt

3. Run the optimization
python main.py

Dependencies

Python 3.7+

numpy

matplotlib

tqdm

Problem Definition: CCOP

Deploy N containers on M servers while optimizing:

Objectives

Minimizing server usage

Minimizing inter-server communication costs

Reducing load imbalance

Constraints

CPU, memory, and storage capacities

Anti-affinity rules

Affinity preferences

Implemented Algorithms
1. Best-Fit with Communication Awareness (Heuristic)

Type: Constructive heuristic

Combines resource utilization and communication cost

Execution time: around 0.1 seconds

Use case: quick feasible solutions

2. Enhanced Tabu Search

Type: single-solution metaheuristic

Features: short-term memory, aspiration criteria, neighborhood sampling

Provides the best improvement (48.4%)

Suitable for high-quality optimization

3. Artificial Immune System

Type: population-based, bio-inspired

Features: clonal selection, hypermutation, diversity maintenance

Balanced trade-off between speed and quality

Usage Examples
Run Full Comparison
from main import main
main()

Custom Problem Instance
from models import ProblemInstance
from heuristics import HeuristicSolver

problem = ProblemInstance(num_containers=100, num_servers=15)
solver = HeuristicSolver(problem)
solution = solver.best_fit_communication_aware()

Configure Metaheuristics
from metaheuristics import EnhancedTabuSearch, ArtificialImmuneCloudOptimizer

tabu_solver = EnhancedTabuSearch(problem, max_iterations=1000, tabu_tenure=50)

immune_solver = ArtificialImmuneCloudOptimizer(
    problem, population_size=100, max_generations=300
)

Output and Analysis

The project generates:

Comparative results across all methods

Convergence curves for metaheuristics

Diversity metrics for population algorithms

Resource utilization analysis

Constraint satisfaction reports

Advanced Features
Cost Function Components

Server usage cost (weight: 1000)

Communication cost (weight: 1)

Load imbalance (weight: 50)

Constraint penalties (weight: 1000)

Constraint Handling

Hard constraints: capacities, anti-affinity

Soft constraints: affinity preferences, load balancing

Feasibility checks at every iteration

Theoretical Background

This project illustrates:

Operations research applied to cloud computing

Combinatorial optimization techniques

Metaheuristic design and implementation

Performance analysis of optimization algorithms
