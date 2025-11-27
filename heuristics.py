import random
from typing import Dict, List
from models import Container, Server, ProblemInstance

class HeuristicSolver:
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.containers = problem.containers
        self.servers = problem.servers
    
    def best_fit_communication_aware(self) -> Dict[int, int]:
        """Heuristique Best-Fit avec prise en compte de la communication"""
        deployment = {}
        server_usage = {
            'cpu': [0.0] * len(self.servers),
            'memory': [0.0] * len(self.servers),
            'storage': [0.0] * len(self.servers)
        }
        
        # Trier les conteneurs par demande de ressources décroissante
        containers_sorted = sorted(self.containers, 
                                 key=lambda x: x.cpu + x.memory, 
                                 reverse=True)
        
        for container in containers_sorted:
            best_server = None
            best_score = float('inf')
            
            for server_id, server in enumerate(self.servers):
                # Vérifier les contraintes de ressources
                if (server_usage['cpu'][server_id] + container.cpu <= server.cpu_capacity and
                    server_usage['memory'][server_id] + container.memory <= server.memory_capacity and
                    server_usage['storage'][server_id] + container.storage <= server.storage_capacity):
                    
                    # Vérifier les contraintes d'anti-affinité
                    if not self._check_anti_affinity(container, server_id, deployment):
                        continue
                    
                    # Calculer le score basé sur la communication avec conteneurs déjà placés
                    comm_score = self._calculate_communication_score(container, server_id, deployment)
                    
                    # Score total : utilisation des ressources + communication
                    cpu_usage = server_usage['cpu'][server_id] / server.cpu_capacity
                    memory_usage = server_usage['memory'][server_id] / server.memory_capacity
                    resource_usage = (cpu_usage + memory_usage) / 2
                    
                    total_score = resource_usage * 0.6 + comm_score * 0.4
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_server = server_id
            
            if best_server is not None:
                deployment[container.id] = best_server
                server_usage['cpu'][best_server] += container.cpu
                server_usage['memory'][best_server] += container.memory
                server_usage['storage'][best_server] += container.storage
            else:
                # Aucun serveur trouvé - affectation aléatoire (devrait être évité)
                available_servers = list(range(len(self.servers)))
                random.shuffle(available_servers)
                deployment[container.id] = available_servers[0]
        
        return deployment
    
    def _check_anti_affinity(self, container: Container, server_id: int, deployment: Dict[int, int]) -> bool:
        """Vérifier les contraintes d'anti-affinité"""
        if container.anti_affinity_group is None:
            return True
        
        for deployed_container_id, deployed_server in deployment.items():
            deployed_container = self.containers[deployed_container_id]
            if (deployed_container.anti_affinity_group == container.anti_affinity_group and
                deployed_server == server_id):
                return False
        return True
    
    def _calculate_communication_score(self, container: Container, server_id: int, deployment: Dict[int, int]) -> float:
        """Calculer le score de communication pour un placement potentiel"""
        comm_score = 0.0
        
        for deployed_container_id, deployed_server in deployment.items():
            if deployed_server == server_id:
                # Communication intra-serveur - pénalité négative (bonne chose)
                comm_cost = self.problem.get_communication_cost(container.id, deployed_container_id)
                comm_score -= comm_cost * 0.1  # Récompense pour communication locale
            else:
                # Communication inter-serveur - pénalité positive
                comm_cost = self.problem.get_communication_cost(container.id, deployed_container_id)
                comm_score += comm_cost
        
        return comm_score
    
    def first_fit_decreasing(self) -> Dict[int, int]:
        """Heuristique First-Fit Decreasing classique"""
        deployment = {}
        server_usage = {
            'cpu': [0.0] * len(self.servers),
            'memory': [0.0] * len(self.servers),
            'storage': [0.0] * len(self.servers)
        }
        
        # Trier les conteneurs par demande de ressources décroissante
        containers_sorted = sorted(self.containers, 
                                 key=lambda x: x.cpu + x.memory, 
                                 reverse=True)
        
        for container in containers_sorted:
            placed = False
            for server_id, server in enumerate(self.servers):
                if (server_usage['cpu'][server_id] + container.cpu <= server.cpu_capacity and
                    server_usage['memory'][server_id] + container.memory <= server.memory_capacity and
                    server_usage['storage'][server_id] + container.storage <= server.storage_capacity and
                    self._check_anti_affinity(container, server_id, deployment)):
                    
                    deployment[container.id] = server_id
                    server_usage['cpu'][server_id] += container.cpu
                    server_usage['memory'][server_id] += container.memory
                    server_usage['storage'][server_id] += container.storage
                    placed = True
                    break
            
            if not placed:
                # Placement d'urgence sur le premier serveur avec assez de capacité
                for server_id, server in enumerate(self.servers):
                    if (server_usage['cpu'][server_id] + container.cpu <= server.cpu_capacity and
                        server_usage['memory'][server_id] + container.memory <= server.memory_capacity):
                        
                        deployment[container.id] = server_id
                        server_usage['cpu'][server_id] += container.cpu
                        server_usage['memory'][server_id] += container.memory
                        break
        
        return deployment