Cloud Container Orchestration Optimizer
1. Project Overview

A framework for solving the Cloud Container Orchestration Problem (CCOP) using heuristics and metaheuristics.
It applies operations research and cloud optimization techniques to efficiently deploy containers on physical servers.

2. Key Features

Heuristic + Metaheuristics (Tabu Search, Artificial Immune System)

Realistic cloud constraints (resource limits, anti-affinity, communication costs)

Performance comparison & visualizations

Modular architecture for easy customization

3. Project Structure
src/
├── models.py              # Container, Server, ProblemInstance classes
├── heuristics.py          # Best-Fit with Communication Awareness heuristic
├── metaheuristics.py      # Tabu Search & Artificial Immune System
├── utils.py               # Evaluation and visualization utilities
├── main.py                # Execution and comparison script
└── requirements.txt       # Dependencies

4. Performance Highlights
Method	Total Cost	Servers Used	Time (s)	Improvement
Heuristic	13,671.82	8 / 10	0.12	Baseline
Tabu Search	7,051.76	6 / 10	7.59	-48.4%
Immune System	9,172.56	8 / 10	5.00	-32.9%
5. Installation
Clone the repository
git clone <repository-url>
cd src

Install dependencies
pip install -r requirements.txt

Run the project
python main.py

6. Dependencies

Python 3.7+

numpy

matplotlib

tqdm

7. Problem Definition
Objectives

Minimize:

Number of servers used

Inter-server communication costs

Load imbalance

Constraints

CPU, memory, storage capacities

Anti-affinity constraints

Affinity preferences

8. Implemented Algorithms
8.1 Best-Fit with Communication Awareness (Heuristic)

Constructive heuristic

Very fast execution (~0.1s)

Produces feasible baseline solutions

8.2 Enhanced Tabu Search

Single-solution metaheuristic

Tabu tenure, aspiration criteria, neighborhood exploration

Best-performing method

8.3 Artificial Immune System

Population-based metaheuristic

Clonal selection, hypermutation, diversity maintenance

Balanced speed/quality

9. Usage Examples
Run all algorithms
from main import main
main()

Custom problem instance
from models import ProblemInstance
from heuristics import HeuristicSolver

problem = ProblemInstance(num_containers=100, num_servers=15)
solver = HeuristicSolver(problem)
solution = solver.best_fit_communication_aware()

Configure metaheuristics
from metaheuristics import EnhancedTabuSearch, ArtificialImmuneCloudOptimizer

tabu_solver = EnhancedTabuSearch(
    problem, max_iterations=1000, tabu_tenure=50
)

immune_solver = ArtificialImmuneCloudOptimizer(
    problem, population_size=100, max_generations=300
)

10. Output and Analysis

The framework generates:

Method comparison reports

Convergence curves

Population diversity metrics

Resource utilization plots

Constraint satisfaction checks

11. Cost Function

The total cost is composed of:

Server usage cost (weight: 1000)

Communication cost (weight: 1)

Load imbalance cost (weight: 50)

Penalties for constraint violations (weight: 1000)

12. Constraint Handling

Hard constraints: capacities, anti-affinity

Soft constraints: affinity preferences, load balancing

Feasibility checks at each iteration

13. Theoretical Background

This project illustrates:

Operations research applied to cloud systems

Combinatorial optimization modeling

Metaheuristic algorithm design

Rigorous performance evaluation
