import time
import json
from typing import Dict

from models import ProblemInstance
from heuristics import HeuristicSolver
from metaheuristics import EnhancedTabuSearch, ArtificialImmuneCloudOptimizer
from utils import SolutionAnalyzer, Visualizer

def main():
    print(">>> OPTIMISATION D'ORCHESTRATION DE CONTENEURS CLOUD")
    print("=" * 60)
    
    # Configuration
    NUM_CONTAINERS = 50
    NUM_SERVERS = 10
    
    # Générer l'instance du problème
    print("Génération de l'instance du problème...")
    problem = ProblemInstance(NUM_CONTAINERS, NUM_SERVERS)
    
    print(f"Problème généré: {NUM_CONTAINERS} conteneurs, {NUM_SERVERS} serveurs")
    print(f"Coûts de communication: {len(problem.communication_cost)} paires")
    
    analyzer = SolutionAnalyzer(problem)
    results = {}
    
    # 1. Heuristique Best-Fit avec Communication
    print("\n" + "="*60)
    print("1. EXÉCUTION DE L'HEURISTIQUE")
    print("="*60)
    
    start_time = time.time()
    heuristic_solver = HeuristicSolver(problem)
    heuristic_solution = heuristic_solver.best_fit_communication_aware()
    heuristic_time = time.time() - start_time
    
    heuristic_stats = analyzer.analyze_solution(heuristic_solution, "Heuristique Best-Fit")
    heuristic_cost = heuristic_solver.problem.evaluate_solution(heuristic_solution)
    results['Heuristique'] = {
        'cost': heuristic_cost,
        'time': heuristic_time,
        'stats': heuristic_stats
    }
    
    # 2. Recherche Tabou
    print("\n" + "="*60)
    print("2. EXÉCUTION DE LA RECHERCHE TABOU")
    print("="*60)
    
    start_time = time.time()
    tabu_solver = EnhancedTabuSearch(problem, max_iterations=500, tabu_tenure=30)
    tabu_solution = tabu_solver.solve()
    tabu_time = time.time() - start_time
    
    tabu_stats = analyzer.analyze_solution(tabu_solution, "Recherche Tabou")
    tabu_cost = tabu_solver.evaluate_solution(tabu_solution)
    results['Recherche Tabou'] = {
        'cost': tabu_cost,
        'time': tabu_time,
        'stats': tabu_stats
    }
    
    # 3. Système Immunitaire Artificiel
    print("\n" + "="*60)
    print("3. EXÉCUTION DU SYSTÈME IMMUNITAIRE ARTIFICIEL")
    print("="*60)
    
    start_time = time.time()
    immune_solver = ArtificialImmuneCloudOptimizer(
        problem, 
        population_size=30,
        max_generations=100,
        clone_factor=0.15
    )
    immune_solution = immune_solver.solve()
    immune_time = time.time() - start_time
    
    immune_stats = analyzer.analyze_solution(immune_solution, "Système Immunitaire")
    immune_cost = immune_solver.evaluate_solution(immune_solution)
    results['Système Immunitaire'] = {
        'cost': immune_cost,
        'time': immune_time,
        'stats': immune_stats
    }
    
    # Affichage des résultats comparatifs
    print("\n" + "="*60)
    print("RÉSULTATS COMPARATIFS")
    print("="*60)
    
    comparison_data = {}
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Coût total: {data['cost']:.2f}")
        print(f"  Temps d'exécution: {data['time']:.2f}s")
        print(f"  Serveurs utilisés: {data['stats']['servers_used']}")
        print(f"  Utilisation CPU moyenne: {data['stats']['avg_cpu_utilization']:.1f}%")
        print(f"  Coût communication: {data['stats']['communication_cost']:.2f}")
        
        comparison_data[method] = data['cost']
    
    # Visualisations
    print("\nGénération des graphiques...")
    
    # Graphique de convergence
    Visualizer.plot_convergence(tabu_solver, immune_solver, comparison_data)
    
    # Graphique de comparaison
    Visualizer.plot_solution_comparison(comparison_data)
    
    # Sauvegarde des résultats
    save_results(results, 'results.json')
    
    print("\n[OK] Analyse terminée!")
    print(f"Amélioration Recherche Tabou vs Heuristique: {((results['Heuristique']['cost'] - results['Recherche Tabou']['cost']) / results['Heuristique']['cost'] * 100):.1f}%")
    print(f"Amélioration Système Immunitaire vs Heuristique: {((results['Heuristique']['cost'] - results['Système Immunitaire']['cost']) / results['Heuristique']['cost'] * 100):.1f}%")

def save_results(results: Dict, filename: str):
    """Sauvegarder les résultats dans un fichier JSON"""
    # Convertir pour la sérialisation JSON
    serializable_results = {}
    for method, data in results.items():
        serializable_results[method] = {
            'cost': float(data['cost']),
            'time': float(data['time']),
            'stats': {
                'servers_used': int(data['stats']['servers_used']),
                'avg_cpu_utilization': float(data['stats']['avg_cpu_utilization']),
                'communication_cost': float(data['stats']['communication_cost'])
            }
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n[SAVE] Résultats sauvegardés dans {filename}")

if __name__ == "__main__":
    main()