Cloud Container Orchestration Optimizer
Project Overview
A comprehensive optimization framework for solving the Cloud Container Orchestration Problem (CCOP) using advanced heuristics and metaheuristics. This project demonstrates the application of operations research techniques to modern cloud computing challenges.

Key Features
Multiple Optimization Approaches: Heuristic, Tabu Search, and Artificial Immune System

Realistic Cloud Constraints: Resource limits, anti-affinity, communication costs

Comprehensive Analysis: Performance comparison and visualization

Modular Design: Easy to extend and customize

Project Structure
text
src/
├── models.py              # Data classes: Container, Server, ProblemInstance
├── heuristics.py          # Best-Fit with Communication Awareness algorithm
├── metaheuristics.py      # Tabu Search and Artificial Immune System
├── utils.py               # Solution analysis, visualization, and utilities
├── main.py                # Main execution and comparison script
└── requirements.txt       # Python dependencies
Performance Highlights
Method	Total Cost	Servers Used	Time (s)	Improvement
Heuristic	13,671.82	8/10	0.12	Baseline
Tabu Search	7,051.76	6/10	7.59	-48.4%
Immune System	9,172.56	8/10	5.00	-32.9%
Installation
Clone the repository

bash
git clone <repository-url>
cd src
Install dependencies

bash
pip install -r requirements.txt
Run the optimization

bash
python main.py
Dependencies
Python 3.7+

numpy

matplotlib

tqdm

Problem Definition
The CCOP Problem
Deploy N containers on M physical servers while:

Minimizing:

Number of servers used

Inter-server communication costs

Load imbalance

Respecting:

CPU, memory, storage constraints

Anti-affinity constraints (containers that must be separated)

Affinity preferences (containers that should be together)

Implemented Algorithms
1. Best-Fit with Communication Awareness (Heuristic)
Type: Constructive heuristic

Strategy: Combines resource utilization and communication costs

Speed: ~0.1 seconds

Use case: Quick feasible solutions

2. Enhanced Tabu Search (Metaheuristic)
Type: Single-solution metaheuristic

Features: Short-term memory, aspiration criteria, neighborhood sampling

Performance: Best results (48.4% improvement)

Use case: High-quality optimization

3. Artificial Immune System (Metaheuristic)
Type: Population-based bio-inspired algorithm

Features: Clonal selection, hypermutation, diversity maintenance

Performance: Good speed/quality balance

Use case: Balanced optimization approach

Usage Examples
Basic Execution
python
from main import main
main()  # Runs complete comparison
Custom Problem Instance
python
from models import ProblemInstance
from heuristics import HeuristicSolver

# Create custom problem
problem = ProblemInstance(num_containers=100, num_servers=15)

# Solve with heuristic
solver = HeuristicSolver(problem)
solution = solver.best_fit_communication_aware()
Algorithm Configuration
python
from metaheuristics import EnhancedTabuSearch, ArtificialImmuneCloudOptimizer

# Configure Tabu Search
tabu_solver = EnhancedTabuSearch(problem, max_iterations=1000, tabu_tenure=50)

# Configure Immune System
immune_solver = ArtificialImmuneCloudOptimizer(
    problem, population_size=100, max_generations=300
)
Output and Analysis
The project generates:

Comparative results between all methods

Convergence plots for metaheuristics

Diversity metrics for population-based algorithms

Resource utilization analysis

Constraint satisfaction reports

Advanced Features
Cost Function
The multi-criteria cost function combines:

Server usage costs (weight: 1000)

Communication costs (weight: 1)

Load imbalance (weight: 50)

Constraint penalties (weight: 1000)

Constraint Handling
Hard constraints: Resource capacities, anti-affinity

Soft constraints: Affinity preferences, load balancing

Feasibility checks in all solution generation

Theoretical Background
This project demonstrates:

Operations Research principles applied to cloud computing

Combinatorial Optimization techniques

Metaheuristic design and implementation

Performance analysis of optimization algorithms
