import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from models import ProblemInstance, Container, Server

class SolutionAnalyzer:
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
    
    def analyze_solution(self, solution: Dict[int, int], method_name: str):
        """Analyser une solution et afficher les statistiques"""
        print(f"\n{'='*50}")
        print(f"ANALYSE - {method_name}")
        print(f"{'='*50}")
        
        # Serveurs utilisés
        servers_used = set(solution.values())
        print(f"Nombre de serveurs utilisés: {len(servers_used)}/{len(self.problem.servers)}")
        
        # Calculer l'utilisation des ressources
        server_usage = self._calculate_server_usage(solution)
        self._print_utilization_stats(server_usage)
        
        # Coût de communication
        comm_cost = self._calculate_communication_cost(solution)
        print(f"Coût total de communication: {comm_cost:.2f}")
        
        # Vérifier les contraintes
        self._check_constraints(solution)
        
        return {
            'servers_used': len(servers_used),
            'avg_cpu_utilization': np.mean([usage['cpu'] for usage in server_usage.values()]),
            'avg_memory_utilization': np.mean([usage['memory'] for usage in server_usage.values()]),
            'communication_cost': comm_cost
        }
    
    def _calculate_server_usage(self, solution: Dict[int, int]) -> Dict[int, Dict]:
        """Calculer l'utilisation des ressources par serveur"""
        server_usage = {}
        
        for server_id in range(len(self.problem.servers)):
            server_usage[server_id] = {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0}
        
        for container_id, server_id in solution.items():
            container = self.problem.containers[container_id]
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
            server_usage[server_id]['storage'] += container.storage
        
        return server_usage
    
    def _print_utilization_stats(self, server_usage: Dict[int, Dict]):
        """Afficher les statistiques d'utilisation"""
        cpu_utilizations = []
        memory_utilizations = []
        
        for server_id, usage in server_usage.items():
            server = self.problem.servers[server_id]
            cpu_util = (usage['cpu'] / server.cpu_capacity) * 100
            memory_util = (usage['memory'] / server.memory_capacity) * 100
            
            cpu_utilizations.append(cpu_util)
            memory_utilizations.append(memory_util)
            
            if usage['cpu'] > 0:  # Afficher seulement les serveurs utilisés
                print(f"Serveur {server_id}: CPU {cpu_util:.1f}%, Mémoire {memory_util:.1f}%")
        
        if cpu_utilizations:
            print(f"\nUtilisation moyenne CPU: {np.mean(cpu_utilizations):.1f}%")
            print(f"Écart-type CPU: {np.std(cpu_utilizations):.1f}%")
            print(f"Utilisation moyenne Mémoire: {np.mean(memory_utilizations):.1f}%")
    
    def _calculate_communication_cost(self, solution: Dict[int, int]) -> float:
        """Calculer le coût total de communication"""
        total_cost = 0.0
        
        for i in range(len(self.problem.containers)):
            for j in range(i + 1, len(self.problem.containers)):
                if solution[i] != solution[j]:  # Communication inter-serveur
                    total_cost += self.problem.get_communication_cost(i, j)
        
        return total_cost
    
    def _check_constraints(self, solution: Dict[int, int]):
        """Vérifier les violations de contraintes"""
        violations = 0
        
        # Vérifier capacité
        server_usage = self._calculate_server_usage(solution)
        for server_id, usage in server_usage.items():
            server = self.problem.servers[server_id]
            if usage['cpu'] > server.cpu_capacity:
                print(f"[WARN] Surcapacité CPU sur serveur {server_id}")
                violations += 1
        
        # Vérifier anti-affinité
        for i in range(len(self.problem.containers)):
            for j in range(i + 1, len(self.problem.containers)):
                container_i = self.problem.containers[i]
                container_j = self.problem.containers[j]
                
                if (container_i.anti_affinity_group is not None and 
                    container_i.anti_affinity_group == container_j.anti_affinity_group and
                    solution[i] == solution[j]):
                    print(f"[WARN] Violation anti-affinité: conteneurs {i} et {j} sur même serveur")
                    violations += 1
        
        if violations == 0:
            print("[OK] Aucune violation de contrainte détectée")
        else:
            print(f"[FAIL] {violations} violation(s) de contrainte détectée(s)")

class Visualizer:
    @staticmethod
    def plot_convergence(tabu_solver, immune_solver, comparison_data=None):
        """Tracer la convergence des algorithmes"""
        plt.figure(figsize=(12, 8))
        
        # Recherche Tabou
        plt.subplot(2, 2, 1)
        plt.plot(tabu_solver.best_cost_history, 'b-', label='Meilleur coût', linewidth=2)
        plt.plot(tabu_solver.current_cost_history, 'r-', alpha=0.5, label='Coût courant')
        plt.title('Convergence - Recherche Tabou')
        plt.xlabel('Itération')
        plt.ylabel('Coût')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Système Immunitaire - Meilleur coût
        plt.subplot(2, 2, 2)
        plt.plot(immune_solver.best_cost_history, 'g-', label='Meilleur coût', linewidth=2)
        plt.plot(immune_solver.avg_cost_history, 'orange', alpha=0.7, label='Coût moyen population')
        plt.title('Convergence - Système Immunitaire')
        plt.xlabel('Génération')
        plt.ylabel('Coût')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Diversité
        plt.subplot(2, 2, 3)
        plt.plot(immune_solver.diversity_history, 'purple', linewidth=2)
        plt.title('Diversité de la Population - Système Immunitaire')
        plt.xlabel('Génération')
        plt.ylabel('Diversité')
        plt.grid(True, alpha=0.3)
        
        # Comparaison finale
        plt.subplot(2, 2, 4)
        plt.title('Comparaison des Méthodes')
        if comparison_data:
            methods = list(comparison_data.keys())
            costs = list(comparison_data.values())
            bars = plt.bar(methods, costs, color=['skyblue', 'lightcoral', 'lightgreen'])
            for bar, cost in zip(bars, costs):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05 * bar.get_height(),
                         f'{cost:.0f}', ha='center', va='bottom')
            plt.ylabel('Coût total')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center', fontsize=12, alpha=0.7)
            plt.xticks([])
            plt.yticks([])
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_solution_comparison(costs: Dict[str, float]):
        """Tracer la comparaison des coûts entre méthodes"""
        plt.figure(figsize=(10, 6))
        
        methods = list(costs.keys())
        cost_values = list(costs.values())
        
        bars = plt.bar(methods, cost_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        # Ajouter les valeurs sur les barres
        for bar, cost in zip(bars, cost_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        plt.title('Comparaison des Coûts par Méthode')
        plt.ylabel('Coût Total')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()