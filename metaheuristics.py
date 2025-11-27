import random
import numpy as np
from typing import Dict, List, Set, Deque
from collections import deque, defaultdict
from tqdm import tqdm

from models import ProblemInstance
from heuristics import HeuristicSolver

class EnhancedTabuSearch:
    def __init__(self, problem: ProblemInstance, max_iterations: int = 1000, tabu_tenure: int = 50):
        self.problem = problem
        self.containers = problem.containers
        self.servers = problem.servers
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list: Deque = deque(maxlen=tabu_tenure)
        
        # Statistiques
        self.best_cost_history = []
        self.current_cost_history = []
    
    def evaluate_solution(self, solution: Dict[int, int]) -> float:
        """Évaluer la qualité d'une solution"""
        total_cost = 0.0
        
        # 1. Coût des serveurs utilisés
        servers_used = len(set(solution.values()))
        total_cost += servers_used * 1000  # Coût fixe par serveur
        
        # 2. Coût de communication inter-serveur
        comm_cost = 0.0
        for i in range(len(self.containers)):
            for j in range(i + 1, len(self.containers)):
                if solution[i] != solution[j]:
                    comm_cost += self.problem.get_communication_cost(i, j)
        
        total_cost += comm_cost
        
        # 3. Pénalité pour déséquilibre de charge
        load_imbalance = self._calculate_load_imbalance(solution)
        total_cost += load_imbalance * 50
        
        # 4. Pénalité pour violations de contraintes
        constraint_penalty = self._calculate_constraint_penalty(solution)
        total_cost += constraint_penalty * 1000
        
        return total_cost
    
    def _calculate_load_imbalance(self, solution: Dict[int, int]) -> float:
        """Calculer le déséquilibre de charge entre serveurs"""
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0})
        
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
        
        cpu_utilizations = [usage['cpu'] / self.servers[server_id].cpu_capacity 
                          for server_id, usage in server_usage.items()]
        memory_utilizations = [usage['memory'] / self.servers[server_id].memory_capacity 
                             for server_id, usage in server_usage.items()]
        
        if not cpu_utilizations:
            return 0.0
            
        cpu_imbalance = np.std(cpu_utilizations)
        memory_imbalance = np.std(memory_utilizations)
        
        return (cpu_imbalance + memory_imbalance) / 2
    
    def _calculate_constraint_penalty(self, solution: Dict[int, int]) -> float:
        """Calculer les pénalités pour violations de contraintes"""
        penalty = 0.0
        
        # Vérifier les contraintes de capacité
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0})
        
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server = self.servers[server_id]
            
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
            server_usage[server_id]['storage'] += container.storage
            
            # Pénalité pour dépassement de capacité
            if server_usage[server_id]['cpu'] > server.cpu_capacity:
                penalty += (server_usage[server_id]['cpu'] - server.cpu_capacity)
            if server_usage[server_id]['memory'] > server.memory_capacity:
                penalty += (server_usage[server_id]['memory'] - server.memory_capacity)
            if server_usage[server_id]['storage'] > server.storage_capacity:
                penalty += (server_usage[server_id]['storage'] - server.storage_capacity)
        
        # Vérifier les contraintes d'anti-affinité
        anti_affinity_violations = 0
        affinity_violations = 0
        
        for i in range(len(self.containers)):
            for j in range(i + 1, len(self.containers)):
                container_i = self.containers[i]
                container_j = self.containers[j]
                
                # Anti-affinité
                if (container_i.anti_affinity_group is not None and 
                    container_i.anti_affinity_group == container_j.anti_affinity_group and
                    solution[i] == solution[j]):
                    anti_affinity_violations += 1
                
                # Affinité (optionnel - pénalité plus faible)
                if (container_i.affinity_group is not None and 
                    container_i.affinity_group == container_j.affinity_group and
                    solution[i] != solution[j]):
                    affinity_violations += 0.5
        
        penalty += anti_affinity_violations + affinity_violations
        
        return penalty
    
    def is_feasible(self, solution: Dict[int, int]) -> bool:
        """Vérifier si une solution est réalisable"""
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0})
        
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server = self.servers[server_id]
            
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
            server_usage[server_id]['storage'] += container.storage
            
            # Vérifier capacité
            if (server_usage[server_id]['cpu'] > server.cpu_capacity or
                server_usage[server_id]['memory'] > server.memory_capacity or
                server_usage[server_id]['storage'] > server.storage_capacity):
                return False
        
        # Vérifier anti-affinité
        for i in range(len(self.containers)):
            for j in range(i + 1, len(self.containers)):
                container_i = self.containers[i]
                container_j = self.containers[j]
                
                if (container_i.anti_affinity_group is not None and 
                    container_i.anti_affinity_group == container_j.anti_affinity_group and
                    solution[i] == solution[j]):
                    return False
        
        return True
    
    def get_neighbors(self, solution: Dict[int, int]) -> List[Dict[int, int]]:
        """Générer les voisins de la solution courante"""
        neighbors = []
        
        # Voisinage : déplacer un conteneur vers un autre serveur
        for container_id in range(len(self.containers)):
            current_server = solution[container_id]
            
            for new_server in range(len(self.servers)):
                if new_server != current_server:
                    neighbor = solution.copy()
                    neighbor[container_id] = new_server
                    neighbors.append(neighbor)
        
        # Échantillonner pour éviter un voisinage trop large
        if len(neighbors) > 100:
            neighbors = random.sample(neighbors, 100)
        
        return neighbors
    
    def solve(self) -> Dict[int, int]:
        """Résoudre le problème avec Recherche Tabou"""
        # Solution initiale avec heuristique
        heuristic_solver = HeuristicSolver(self.problem)
        current_solution = heuristic_solver.best_fit_communication_aware()
        best_solution = current_solution.copy()
        best_cost = self.evaluate_solution(best_solution)
        
        print(f"Solution initiale - Coût: {best_cost:.2f}")
        
        for iteration in tqdm(range(self.max_iterations), desc="Recherche Tabou"):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor in neighbors:
                neighbor_key = tuple(sorted(neighbor.items()))
                
                if neighbor_key not in self.tabu_list or self.evaluate_solution(neighbor) < best_cost:
                    cost = self.evaluate_solution(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
            
            if best_neighbor:
                current_solution = best_neighbor
                self.tabu_list.append(tuple(sorted(current_solution.items())))
                
                current_cost = self.evaluate_solution(current_solution)
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    print(f"Itération {iteration}: Nouveau meilleur coût = {best_cost:.2f}")
            
            # Historique
            self.best_cost_history.append(best_cost)
            self.current_cost_history.append(self.evaluate_solution(current_solution))
            
            # Critère d'arrêt prématuré
            if iteration > 100 and len(set(self.best_cost_history[-50:])) == 1:
                print(f"Convergence détectée à l'itération {iteration}")
                break
        
        return best_solution

class ArtificialImmuneCloudOptimizer:
    def __init__(self, problem: ProblemInstance, population_size: int = 50, 
                 max_generations: int = 200, clone_factor: float = 0.1):
        self.problem = problem
        self.containers = problem.containers
        self.servers = problem.servers
        self.population_size = population_size
        self.max_generations = max_generations
        self.clone_factor = clone_factor
        self.antibodies: List[Dict[int, int]] = []  # Population de solutions
        
        # Statistiques
        self.best_cost_history = []
        self.avg_cost_history = []
        self.diversity_history = []
    
    def initialize_population(self):
        """Initialiser la population avec diversité"""
        heuristic_solver = HeuristicSolver(self.problem)
        
        # Ajouter quelques solutions heuristiques
        for _ in range(5):
            antibody = heuristic_solver.best_fit_communication_aware()
            self.antibodies.append(antibody)
        
        # Ajouter des solutions aléatoires mais réalisables
        for _ in range(self.population_size - 5):
            antibody = self._generate_random_antibody()
            self.antibodies.append(antibody)
    
    def _generate_random_antibody(self) -> Dict[int, int]:
        """Générer une solution aléatoire réalisable"""
        antibody = {}
        server_usage = {
            'cpu': [0.0] * len(self.servers),
            'memory': [0.0] * len(self.servers),
            'storage': [0.0] * len(self.servers)
        }
        
        # Trier les conteneurs pour placer les plus gros d'abord
        containers_sorted = sorted(self.containers, 
                                 key=lambda x: x.cpu + x.memory, 
                                 reverse=True)
        
        for container in containers_sorted:
            feasible_servers = []
            
            for server_id, server in enumerate(self.servers):
                if (server_usage['cpu'][server_id] + container.cpu <= server.cpu_capacity and
                    server_usage['memory'][server_id] + container.memory <= server.memory_capacity and
                    server_usage['storage'][server_id] + container.storage <= server.storage_capacity):
                    
                    # Vérifier anti-affinité
                    anti_affinity_ok = True
                    for deployed_id, deployed_server in antibody.items():
                        deployed_container = self.containers[deployed_id]
                        if (deployed_container.anti_affinity_group is not None and
                            deployed_container.anti_affinity_group == container.anti_affinity_group and
                            deployed_server == server_id):
                            anti_affinity_ok = False
                            break
                    
                    if anti_affinity_ok:
                        feasible_servers.append(server_id)
            
            if feasible_servers:
                chosen_server = random.choice(feasible_servers)
                antibody[container.id] = chosen_server
                server_usage['cpu'][chosen_server] += container.cpu
                server_usage['memory'][chosen_server] += container.memory
                server_usage['storage'][chosen_server] += container.storage
            else:
                # Placement d'urgence
                antibody[container.id] = random.randint(0, len(self.servers) - 1)
        
        return antibody
    
    def evaluate_solution(self, solution: Dict[int, int]) -> float:
        """Évaluer la qualité d'une solution (identique à Tabu Search)"""
        total_cost = 0.0
        
        # Coût des serveurs utilisés
        servers_used = len(set(solution.values()))
        total_cost += servers_used * 1000
        
        # Coût de communication
        comm_cost = 0.0
        for i in range(len(self.containers)):
            for j in range(i + 1, len(self.containers)):
                if solution[i] != solution[j]:
                    comm_cost += self.problem.get_communication_cost(i, j)
        
        total_cost += comm_cost
        
        # Pénalité pour déséquilibre
        load_imbalance = self._calculate_load_imbalance(solution)
        total_cost += load_imbalance * 50
        
        # Pénalité pour violations
        constraint_penalty = self._calculate_constraint_penalty(solution)
        total_cost += constraint_penalty * 1000
        
        return total_cost
    
    def _calculate_load_imbalance(self, solution: Dict[int, int]) -> float:
        """Calculer le déséquilibre de charge"""
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0})
        
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
        
        cpu_utilizations = [usage['cpu'] / self.servers[server_id].cpu_capacity 
                          for server_id, usage in server_usage.items()]
        
        if not cpu_utilizations:
            return 0.0
            
        return np.std(cpu_utilizations)
    
    def _calculate_constraint_penalty(self, solution: Dict[int, int]) -> float:
        """Calculer les pénalités pour violations de contraintes"""
        penalty = 0.0
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0})
        
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server = self.servers[server_id]
            
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
            server_usage[server_id]['storage'] += container.storage
            
            if server_usage[server_id]['cpu'] > server.cpu_capacity:
                penalty += (server_usage[server_id]['cpu'] - server.cpu_capacity)
        
        return penalty
    
    def affinity(self, antibody1: Dict[int, int], antibody2: Dict[int, int]) -> float:
        """Calculer l'affinité (similarité) entre deux solutions"""
        common_assignments = 0
        for container_id in range(len(self.containers)):
            if antibody1[container_id] == antibody2[container_id]:
                common_assignments += 1
        return common_assignments / len(self.containers)
    
    def clonal_selection(self) -> List[Dict[int, int]]:
        """Sélection et clonage des meilleurs anticorps"""
        # Trier par qualité
        ranked_antibodies = sorted(self.antibodies, 
                                 key=lambda x: self.evaluate_solution(x))
        
        # Sélectionner les meilleurs
        selection_size = max(5, int(self.population_size * 0.2))
        best_antibodies = ranked_antibodies[:selection_size]
        
        clones = []
        
        for i, antibody in enumerate(best_antibodies):
            # Nombre de clones inversement proportionnel au rang
            num_clones = max(1, int(self.population_size * self.clone_factor * (1 - i/selection_size)))
            
            for _ in range(num_clones):
                clone = antibody.copy()
                clone = self.mutate_antibody(clone)
                clones.append(clone)
        
        return clones
    
    def mutate_antibody(self, antibody: Dict[int, int]) -> Dict[int, int]:
        """Mutation adaptative d'un anticorps"""
        mutation_rate = 0.3  # Taux de mutation élevé
        
        for container_id in range(len(self.containers)):
            if random.random() < mutation_rate:
                # Changer le serveur de ce conteneur
                current_server = antibody[container_id]
                new_server = random.randint(0, len(self.servers) - 1)
                
                # Éviter les mutations qui créent des violations graves
                if new_server != current_server:
                    antibody[container_id] = new_server
        
        return antibody
    
    def calculate_diversity(self, population: List[Dict[int, int]]) -> float:
        """Calculer la diversité de la population"""
        if len(population) <= 1:
            return 0.0
        
        total_affinity = 0.0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_affinity += self.affinity(population[i], population[j])
                count += 1
        
        return 1.0 - (total_affinity / count) if count > 0 else 0.0
    
    def immune_response(self, new_antibodies: List[Dict[int, int]]) -> List[Dict[int, int]]:
        """Maintenir la diversité de la population"""
        # Trier par qualité
        new_antibodies_sorted = sorted(new_antibodies, 
                                     key=lambda x: self.evaluate_solution(x))
        
        diverse_population = []
        similarity_threshold = 0.8
        
        for antibody in new_antibodies_sorted:
            is_diverse = True
            
            for existing in diverse_population:
                if self.affinity(antibody, existing) > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse or len(diverse_population) < self.population_size // 2:
                diverse_population.append(antibody)
            
            if len(diverse_population) >= self.population_size:
                break
        
        # Compléter avec de nouveaux anticorps si nécessaire
        while len(diverse_population) < self.population_size:
            new_antibody = self._generate_random_antibody()
            diverse_population.append(new_antibody)
        
        return diverse_population
    
    def solve(self) -> Dict[int, int]:
        """Résoudre le problème avec le système immunitaire artificiel"""
        print("Initialisation de la population...")
        self.initialize_population()
        
        best_solution = min(self.antibodies, key=lambda x: self.evaluate_solution(x))
        best_cost = self.evaluate_solution(best_solution)
        
        print(f"Meilleure solution initiale - Coût: {best_cost:.2f}")
        
        for generation in tqdm(range(self.max_generations), desc="Système Immunitaire"):
            # Clonage des meilleurs
            clones = self.clonal_selection()
            
            # Mutation des clones
            mutated_clones = [self.mutate_antibody(clone) for clone in clones]
            
            # Évaluation et sélection
            all_candidates = self.antibodies + mutated_clones
            all_candidates_sorted = sorted(all_candidates, 
                                         key=lambda x: self.evaluate_solution(x))
            
            # Réponse immunitaire pour maintenir la diversité
            self.antibodies = self.immune_response(all_candidates_sorted[:self.population_size * 2])
            
            # Mettre à jour la meilleure solution
            current_best = min(self.antibodies, key=lambda x: self.evaluate_solution(x))
            current_best_cost = self.evaluate_solution(current_best)
            
            if current_best_cost < best_cost:
                best_solution = current_best.copy()
                best_cost = current_best_cost
                print(f"Génération {generation}: Nouveau meilleur coût = {best_cost:.2f}")
            
            # Enregistrer les statistiques
            self.best_cost_history.append(best_cost)
            avg_cost = np.mean([self.evaluate_solution(ab) for ab in self.antibodies])
            self.avg_cost_history.append(avg_cost)
            self.diversity_history.append(self.calculate_diversity(self.antibodies))
        
        return best_solution