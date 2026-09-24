import random, os, sys, copy
from tabnanny import check
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from job import Job, Replica
from collections import deque

class GeneticSearch:
    def __init__(self, master_node, population_size=20, crossover_rate=0.8, mutation_rate=0.5):
        self.master_node = master_node
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def start(self, job, master_node, nb_iteration, threshold, heterogeneous, t_reference):
        free_nodes = master_node.node_to_use_to_starts(job.nb_tasks, job, heterogeneous= heterogeneous)

        # --- Initial population ---
        population = []
        i = 0
        while i < self.population_size:
            ind = self.random_solution(free_nodes)
            bool, _ = checkUtility(master_node, job, ind, t_reference, threshold)
            if bool:
                i +=1
                population.append(ind)

        best_solution = []
        best_flow_time = float('-inf')
        
        for _ in range(nb_iteration):
            # --- Evaluation and filter invalid solution ---
            evaluated = []
            for ind in population:
                #print("Start searching1")
                ind = self.random_solution(free_nodes)
                bool, nodes_and_replicas = checkUtility(master_node, job, ind, t_reference, threshold)
                if bool:
                    evaluated.append((ind,estimateFlowTime(master_node, job, ind,nodes_and_replicas,t_reference)))


            # If no valid solution remains → regenerate
            if len(evaluated)  < 2:
                i = 0
                while i < self.population_size:
                    ind = self.random_solution(free_nodes)
                    bool, _ = checkUtility(master_node, job, ind, t_reference, threshold)
                    if bool:
                        i +=1
                        population.append(ind)
                continue

            # Update the best solution found so far
            for ind, util in evaluated:
                if util < best_flow_time:
                    best_solution, best_flow_time = ind, util

            # --- Selection (elitism) ---
            evaluated.sort(key=lambda x: x[1], reverse=True)
            selected = [ind for ind, _ in evaluated[:self.population_size // 2]]  # keep top 50%
            
            # --- Crossover ---
            children = []
            while len(children) < self.population_size - len(selected) and len(selected) > 1:
                if random.random() < self.crossover_rate:
                    p1, p2 = random.sample(selected, 2)
                    c1, c2 = self.crossover(p1, p2, free_nodes)
                    children.extend([c1, c2])
                """ else:
                    children.append(random.choice(selected))"""

            # --- Mutation ---
            new_population = selected + [
                self.mutate(child, free_nodes) if random.random() < self.mutation_rate else child
                for child in children
            ]
            
            population = new_population

        return best_solution, best_flow_time

    # --- Generate a random solution ---
    def random_solution(self, free_nodes):
        nb_nodes = random.randint(0, len(free_nodes))
        return random.sample(free_nodes, nb_nodes) if nb_nodes > 0 else []

    # --- Crossover between two solutions ---
    def crossover(self, parent1, parent2, free_nodes):
        cut = len(parent1) // 2
        p1 = copy.copy(parent1)
        p2 = copy.copy(parent2)
        child1 = list(p1[:cut] + p2[cut:])
        child2 = list(p2[:cut] + p1[cut:])
        return child1, child2

    # --- Mutation: add / remove / modify a node ---
    def mutate(self, solution, free_nodes):
        solution = solution[:]
        operation = random.choice(["add", "remove", "modify"])

        if operation == "remove" and solution:
            solution.remove(random.choice(solution))

        elif operation == "add":
            available = []
            for node in free_nodes:
                if node not in solution:
                    available.append(node)
            if available:
                solution.append(random.choice(available))

        elif operation == "modify" and solution:
            to_replace = random.choice(solution)
            available = []
            for node in free_nodes:
                if node not in solution:
                    available.append(node)
            if available:
                new_node = random.choice(available)
                solution.remove(to_replace)
                solution.append(new_node)

        return solution

class RandomSearch:
    def __init__(self, master_node):
        self.master_node = master_node


    def start(self, job, master_node, nb_iteration, threshold, heterogeneous, t_reference):
        free_nodes = master_node.node_to_use_to_starts(job.nb_tasks, job, heterogeneous=heterogeneous)
        min_flow_time = float('inf')
        best_solution = []

        for i in range(nb_iteration):
            nb_nodes = random.randint(0, len(free_nodes))
            
            selected_nodes = random.sample(free_nodes, nb_nodes) if nb_nodes > 0 else [] 

            solution_utility,replicas_and_tasks = checkUtilityProblem(master_node, job, selected_nodes, t_reference)

            if solution_utility:
                #return selected_nodes, 0

                solution_flow_time = estimateFlowTime(master_node, job, selected_nodes, replicas_and_tasks,t_reference)
                if solution_flow_time < min_flow_time:
                    best_solution = selected_nodes
                    min_flow_time = solution_flow_time

        return best_solution, float('-inf')

class TabuSearch:
    def __init__(self, master_node, tabu_size=50, neighborhood_size=50):
        self.master_node = master_node
        self.tabu_size = tabu_size           # Maximum number of tabu solutions stored
        self.neighborhood_size = neighborhood_size  # Number of neighbors generated per iteration

    def start(self, job, master_node, nb_iteration, threshold, heterogeneous, t_reference):
        free_nodes = master_node.node_to_use_to_starts(job.nb_tasks, job, heterogeneous=heterogeneous)
        
        t_reference = job.tasks[0].duration
        # --- Initialization ---
        current_solution = self.random_solution(free_nodes)
        best_solution = current_solution
        min_flow_time = float('inf')

        # --- Tabu list (FIFO queue) ---
        tabu_list = deque(maxlen=self.tabu_size)

        # --- Evaluate initial solution ---
        solution_utility, replicas_and_tasks = checkUtilityProblem(master_node, job, current_solution, t_reference)
        if solution_utility:
            min_flow_time = estimateFlowTime(master_node, job, current_solution, replicas_and_tasks, t_reference)

        # --- Main loop ---
        for _ in range(nb_iteration):
            neighborhood = self.generate_neighbors(current_solution, free_nodes)
            best_neighbor = None
            best_neighbor_flow = float('inf')

            # Evaluate each neighbor
            for neighbor in neighborhood:
                if tuple(neighbor) in tabu_list:
                    continue  # Skip tabu solutions

                solution_utility, replicas_and_tasks = checkUtilityProblem(master_node, job, neighbor, t_reference)
                if not solution_utility:
                    continue

                flow_time = estimateFlowTime(master_node, job, neighbor, replicas_and_tasks, t_reference)
                # Keep the best valid neighbor
                if flow_time < best_neighbor_flow:
                    best_neighbor = neighbor
                    best_neighbor_flow = flow_time
            
            # If no valid neighbor found, skip iteration
            if best_neighbor is None:
                continue
            
            # Move to the best neighbor
            current_solution = best_neighbor
            tabu_list.append(tuple(best_neighbor))  # Store as immutable

            # Update global best
            if best_neighbor_flow < min_flow_time:
                best_solution = best_neighbor
                min_flow_time = best_neighbor_flow

        return best_solution, min_flow_time

    # --- Generate an initial random solution ---
    def random_solution(self, free_nodes):
        nb_nodes = random.randint(1, len(free_nodes))
        return random.sample(free_nodes, nb_nodes)

    # --- Generate neighbors by modifying current solution ---
    def generate_neighbors(self, solution, free_nodes):
        neighbors = []
        for _ in range(self.neighborhood_size):
            new_solution = solution[:]
            operation = random.choice(["add", "remove", "swap"])

            if operation == "add":
                available = self.getAvailable(free_nodes, new_solution)
                if available:
                    new_solution.append(random.choice(available))

            elif operation == "remove" and new_solution:
                new_solution.remove(random.choice(new_solution))

            elif operation == "swap" and new_solution:
                available = self.getAvailable(free_nodes, new_solution)
                if available:
                    to_replace = random.choice(new_solution)
                    new_node = random.choice(available)
                    new_solution.remove(to_replace)
                    new_solution.append(new_node)

            # Avoid empty solutions
            if new_solution and new_solution not in neighbors:
                neighbors.append(new_solution)

        return neighbors

    def getAvailable(self, free_nodes, solution):
        available = []
        for node in free_nodes:
            if node not in solution:
                available.append(node)
        return available

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw

def checkUtility(master_node, job, nodes_sample, t_reference):
    replicas_to_add = []
    all_excuted_task = 0
    acceptable_node = []
    
    t_now = master_node.env.now
    nodes_and_replicas = {}

    if len(nodes_sample) == 0:
        return True, nodes_and_replicas

    for node_info in range(len(nodes_sample)):
        
        transfer_time = transferCost(master_node,job.dataset_size, nodes_sample[node_info][0].bandwidth)+1
        new_replicas = Replica(job.job_id, node_id=nodes_sample[node_info][0].node_id, data_size=job.dataset_size,transfer_time=transfer_time, transfer_start_time=t_now) 
        
        all_excuted_task = 0
        replicas_to_check = sortReplicasByNodesBw(job.replicas + replicas_to_add + [new_replicas], master_node)

        nodes_and_replicas = {}
        
        for i,rep in enumerate(replicas_to_check):
            all_excuted_task = 0
            rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[rep.node_id].bandwidth)+1
            for a_rep in replicas_to_check[0:i]:
                a_rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[a_rep.node_id].bandwidth)+1
                nb_task = ((rep.transfer_start_time + rep_transfer_time) - (a_rep.transfer_start_time + a_rep_transfer_time))/(t_reference * master_node.compute_nodes[a_rep.node_id].compute_capacity)+1
                if 1 > nb_task > 0: nb_task = 1
                else: nb_task  = int(nb_task)+1 
                all_excuted_task = all_excuted_task + nb_task
                nodes_and_replicas[a_rep.node_id] = nb_task
                if all_excuted_task > job.nb_tasks:
                    return False,  {}
            
        remaining_tasks = job.nb_tasks - all_excuted_task
        
        if remaining_tasks <= 0:
            return False,  {}
        
        replicas_to_check = sortReplicasByNodesVCPU(job.replicas + replicas_to_add + [new_replicas], master_node)

        for i,rep in enumerate(replicas_to_check):
            nb_task_needed = int(rep.transfer_time / (master_node._config['threshold']*t_reference*master_node.compute_nodes[rep.node_id].compute_capacity))+1
            nb_task_to_compelet = nb_task_needed - (nodes_and_replicas[rep.node_id] if rep.node_id in nodes_and_replicas.keys() else 0)
            t_to_finish = (nb_task_to_compelet * t_reference * master_node.compute_nodes[rep.node_id].compute_capacity)
            rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[rep.node_id].bandwidth)+1
            nb_task_executed = 0
            for replicas_befor in replicas_to_check[0:i]:
                """
                if rep.node_id not in nodes_and_replicas.keys():
                    nb_task_executed += (int(t_to_finish)/(t_reference * master_node.compute_nodes[replicas_befor.node_id].compute_capacity)+1)
                else:
                """
                nb_task = (int(t_to_finish)/(t_reference * master_node.compute_nodes[replicas_befor.node_id].compute_capacity)+1)
                
                nb_task_executed += nb_task
                
                if remaining_tasks <= nb_task_executed or (remaining_tasks - nb_task_executed < nb_task_to_compelet):
                    return False,  {}
                else:
                    nodes_and_replicas[rep.node_id] = nb_task_needed

            remaining_tasks = remaining_tasks - nb_task_to_compelet

            if remaining_tasks < 0:
                return False,  {}
            else:
                nodes_and_replicas[rep.node_id] = nb_task_needed

    return True, nodes_and_replicas

def estimateFlowTimeV1(master_node, job, nodes_sample, replicas_and_tasks,t_reference):
    
    max_replica_flow_time = 0
    
    """if len(nodes_sample) == 0:
        return float('inf')"""
    
    """for replicas in job.replicas:
        transfer_time = transferCost(master_node,job.dataset_size, master_node.compute_nodes[replicas.node_id].bandwidth)
        min_nb_task = int(transfer_time / (master_node._config['threshold']*t_reference*master_node.compute_nodes[replicas.node_id].compute_capacity))+1
        replica_flow_time = (t_now - job.arriving_time) + transfer_time + (min_nb_task * t_reference * master_node.compute_nodes[replicas.node_id].compute_capacity)
        if replica_flow_time > max_replica_flow_time:
            max_replica_flow_time = replica_flow_time"""
    
    t_now = master_node.env.now
    for node_info in range(len(nodes_sample)):
        if nodes_sample[node_info][0].node_id not in replicas_and_tasks.keys() : continue
        
        transfer_time = transferCost(master_node,job.dataset_size, nodes_sample[node_info][0].bandwidth)
        replica_flow_time = (t_now - job.arriving_time) + transfer_time + (replicas_and_tasks[nodes_sample[node_info][0].node_id] * t_reference * master_node.compute_nodes[nodes_sample[node_info][0].node_id].compute_capacity)
        if replica_flow_time > max_replica_flow_time:
            max_replica_flow_time = replica_flow_time
    
    return max_replica_flow_time

def estimateFlowTime(master_node, job, nodes_sample, replicas_and_tasks,t_reference):
    
    max_replica_flow_time = 0
    
    for rep in job.replicas:
        rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[rep.node_id].bandwidth)
        replica_flow_time = (rep.transfer_start_time - job.arriving_time) + rep_transfer_time + (replicas_and_tasks[rep.node_id] * t_reference * master_node.compute_nodes[rep.node_id].compute_capacity)
        if replica_flow_time > max_replica_flow_time:
            max_replica_flow_time = replica_flow_time
    
    t_now = master_node.env.now
    for node_info in range(len(nodes_sample)):
        if nodes_sample[node_info][0].node_id not in replicas_and_tasks.keys() : continue
        
        transfer_time = transferCost(master_node,job.dataset_size, nodes_sample[node_info][0].bandwidth)
        replica_flow_time = (t_now - job.arriving_time) + transfer_time + (replicas_and_tasks[nodes_sample[node_info][0].node_id] * t_reference * master_node.compute_nodes[nodes_sample[node_info][0].node_id].compute_capacity)
        if replica_flow_time > max_replica_flow_time:
            max_replica_flow_time = replica_flow_time
    
    return max_replica_flow_time


def sortReplicasByNodesBw(replica_list, master_node):
    return sorted(replica_list, key=lambda r: (r.transfer_start_time + transferCost(master_node, r.data_size, master_node.compute_nodes[r.node_id].bandwidth)), reverse=False)

def sortReplicasByNodesVCPU(replica_list, master_node):
    return sorted(replica_list, key=lambda r: master_node.compute_nodes[r.node_id].compute_capacity, reverse=False)

def checkUtilityProblem(master_node, job:Job,free_nodes, t_reference= None):
    replicas_to_add = []
    acceptable_node = []
    all_excuted_task = 0

    t_now = master_node.env.now
    
    nodes_and_replicas = {}

    for node_info in range(len(free_nodes)):

        transfer_time = transferCost(master_node, job.dataset_size, free_nodes[node_info][0].bandwidth)
        new_replicas = Replica(job.job_id, node_id=free_nodes[node_info][0].node_id, data_size=job.dataset_size,transfer_time=transfer_time, transfer_start_time=t_now)

        all_excuted_task = 0
        replicas_to_check = sortReplicasByNodesBw(job.replicas + replicas_to_add + [new_replicas], master_node)

        nodes_and_replicas = {}
        a_rep = replicas_to_check[-1]
        a_rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[a_rep.node_id].bandwidth)
        nodes_and_replicas[a_rep.node_id] = 0

        for i,rep in enumerate(replicas_to_check[0:-1]):
            rep_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[rep.node_id].bandwidth)

            nb_task = ((a_rep.transfer_start_time + a_rep_transfer_time) - (rep.transfer_start_time + rep_transfer_time))/(t_reference * master_node.compute_nodes[rep.node_id].compute_capacity)+1

            if 1 > nb_task > 0: nb_task = 1
            else: nb_task  = int(nb_task)+1 

            all_excuted_task = all_excuted_task + nb_task
            nodes_and_replicas[rep.node_id] = nb_task

            if all_excuted_task > job.nb_tasks:
                return False,  {}

        remaining_tasks = job.nb_tasks - all_excuted_task
        
        if remaining_tasks <= 0:
            return False, {}

        replicas_to_check = sortReplicasByNodesVCPU(job.replicas + replicas_to_add+ [new_replicas], master_node)

        max_makespan, nb_task_per_nodes = master_node.alloc_branch_and_bound(remaining_tasks, replicas_to_check)
        add = True
        for i,nb_task_per_node in enumerate(nb_task_per_nodes):
            replica = replicas_to_check[i]
            replica_transfer_time = transferCost(master_node, job.dataset_size, master_node.compute_nodes[replica.node_id].bandwidth)
            nodes_and_replicas[replica.node_id] = nodes_and_replicas[replica.node_id]+nb_task_per_node
            if nb_task_per_node+nodes_and_replicas[replica.node_id] > 0 and replica_transfer_time / ((nodes_and_replicas[replica.node_id]) * t_reference * master_node.compute_nodes[replica.node_id].compute_capacity) > master_node.threshold:
                add = False
                break
            elif nb_task_per_node+nodes_and_replicas[replica.node_id] < 0:
                add = False
                break 
        
        if add:
            acceptable_node.append(free_nodes[node_info][0].node_id)
        else:
            return False,  {}
    
    return True, nodes_and_replicas
