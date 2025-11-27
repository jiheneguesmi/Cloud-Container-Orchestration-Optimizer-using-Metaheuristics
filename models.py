import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from statistics import pstdev

@dataclass
class Container:
    id: int
    cpu: float
    memory: float
    storage: float
    affinity_group: int = None  # Conteneurs devant être ensemble
    anti_affinity_group: int = None  # Conteneurs devant être séparés

@dataclass
class Server:
    id: int
    cpu_capacity: float
    memory_capacity: float
    storage_capacity: float
    cost: float = 1.0

class ProblemInstance:
    def __init__(self, num_containers: int, num_servers: int):
        self.containers: List[Container] = []
        self.servers: List[Server] = []
        self.communication_cost: Dict[Tuple[int, int], float] = {}
        self.num_containers = num_containers
        self.num_servers = num_servers
        
        self._generate_instance()
    
    def _generate_instance(self):
        # Générer les serveurs
        for i in range(self.num_servers):
            server = Server(
                id=i,
                cpu_capacity=random.uniform(8, 32),
                memory_capacity=random.uniform(16, 64),
                storage_capacity=random.uniform(100, 500)
            )
            self.servers.append(server)
        
        # Générer les conteneurs
        for i in range(self.num_containers):
            container = Container(
                id=i,
                cpu=random.uniform(0.5, 4),
                memory=random.uniform(1, 8),
                storage=random.uniform(5, 20),
                affinity_group=random.choice([None, 1, 2]) if random.random() < 0.3 else None,
                anti_affinity_group=random.choice([None, 1, 2]) if random.random() < 0.2 else None
            )
            self.containers.append(container)
        
        # Générer les coûts de communication
        for i in range(self.num_containers):
            for j in range(i + 1, self.num_containers):
                # Plus de communication entre certains groupes de conteneurs
                base_cost = random.uniform(0.1, 2.0)
                if (self.containers[i].affinity_group is not None and 
                    self.containers[i].affinity_group == self.containers[j].affinity_group):
                    base_cost *= 5  # Communication importante entre conteneurs affines
                
                self.communication_cost[(i, j)] = base_cost
                self.communication_cost[(j, i)] = base_cost
    
    def get_communication_cost(self, container1: int, container2: int) -> float:
        return self.communication_cost.get((container1, container2), 0.0)

    def evaluate_solution(self, solution: Dict[int, int]) -> float:
        """Évaluer une solution en combinant coûts et pénalités."""
        total_cost = 0.0

        # Coût fixe par serveur utilisé
        servers_used = len(set(solution.values()))
        total_cost += servers_used * 1000

        # Coût de communication inter-serveur
        comm_cost = 0.0
        for i in range(self.num_containers):
            for j in range(i + 1, self.num_containers):
                if solution.get(i) is None or solution.get(j) is None:
                    continue
                if solution[i] != solution[j]:
                    comm_cost += self.get_communication_cost(i, j)
        total_cost += comm_cost

        # Calcul des utilisations par serveur
        server_usage = defaultdict(lambda: {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0})
        for container_id, server_id in solution.items():
            container = self.containers[container_id]
            server_usage[server_id]['cpu'] += container.cpu
            server_usage[server_id]['memory'] += container.memory
            server_usage[server_id]['storage'] += container.storage

        # Pénalité de déséquilibre de charge
        cpu_utils = [
            usage['cpu'] / self.servers[server_id].cpu_capacity
            for server_id, usage in server_usage.items()
            if self.servers[server_id].cpu_capacity > 0
        ]
        mem_utils = [
            usage['memory'] / self.servers[server_id].memory_capacity
            for server_id, usage in server_usage.items()
            if self.servers[server_id].memory_capacity > 0
        ]

        cpu_std = pstdev(cpu_utils) if len(cpu_utils) >= 2 else 0.0
        mem_std = pstdev(mem_utils) if len(mem_utils) >= 2 else 0.0
        load_imbalance = (cpu_std + mem_std) / 2
        total_cost += load_imbalance * 50

        # Pénalités de contraintes
        constraint_penalty = 0.0
        for server_id, usage in server_usage.items():
            server = self.servers[server_id]
            if usage['cpu'] > server.cpu_capacity:
                constraint_penalty += usage['cpu'] - server.cpu_capacity
            if usage['memory'] > server.memory_capacity:
                constraint_penalty += usage['memory'] - server.memory_capacity
            if usage['storage'] > server.storage_capacity:
                constraint_penalty += usage['storage'] - server.storage_capacity

        for i in range(self.num_containers):
            for j in range(i + 1, self.num_containers):
                container_i = self.containers[i]
                container_j = self.containers[j]

                if container_i.anti_affinity_group is not None and \
                   container_i.anti_affinity_group == container_j.anti_affinity_group and \
                   solution.get(i) == solution.get(j):
                    constraint_penalty += 1

                if container_i.affinity_group is not None and \
                   container_i.affinity_group == container_j.affinity_group and \
                   solution.get(i) is not None and solution.get(j) is not None and \
                   solution[i] != solution[j]:
                    constraint_penalty += 0.5

        total_cost += constraint_penalty * 1000

        return total_cost