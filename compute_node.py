import logging
import simpy
import numpy as np
from utils.classifier import estimatingParams

logger = logging.getLogger(__name__)

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw

class ComputeNode:
    def __init__(self, env, node_id, master, bandwidth,compute_capacity=1, energy_consumption=1, capacity=float('inf')):
        self.env = env
        self.node_id = node_id
        self.queue = simpy.Store(env)
        self.master = master
        self.bandwidth = bandwidth
        self.bandwidth_lock = simpy.Resource(env, capacity=bandwidth)
        self.list_of_datasets = []
        self.free_node = True
        self.node_ready_event = None
        self.running_task = []
        self.running_transfer = False
        self.running_job = []
        self.datasets = []
        self.job_order = []
        self.compute_capacity:float = compute_capacity if master._config['homogeneous'] == False else 1
        self.energy_consumption:float = energy_consumption
        self.link_occuped = False
        self.new_task = None

        self.replicas = []
        # Storage capacity in MB (same unit as job.dataset_size). float('inf') means unconstrained.
        self.capacity:float = capacity
        # job_id of the dataset currently occupying this node's single storage slot, or None.
        self.current_dataset = None
        # Inherited from the full repo, where these two fields track an eviction in
        # progress (preemption/migration - removed from this artifact, see
        # heuristic_scheduler.py): never set here, only their default value (None) is
        # ever used.
        self.reserved_by = None
        self.pending_departure = None

    def process_tasks(self):
        self.node_ready_event = self.env.event()
        while True:
            self.free_node = True
            self.node_ready_event.succeed()
            new_task = yield self.queue.get()
            self.free_node = False
            self.node_ready_event = self.env.event()
            logger.debug("[%s] Compute-%s: Got new task with job_id: %s, task_id: %s, duration: %s, dataset_size: %s",
                         self.env.now, self.node_id, new_task.job_id, new_task.task_id, new_task.duration, new_task.dataset_size)
            # Wait for the data transfer event to complete
            yield new_task.dataset_ready_event
            # Simulate task processing
            start_time = self.env.now
            new_task.status = "Started"
            new_task.node= self.node_id
            new_task.starting_time = start_time
            self.master.tracker.log_task_start(new_task.job_id, new_task.task_id, self.node_id, start_time)
            yield self.env.timeout(new_task.duration)  # actual processing
            end_time = self.env.now
            #new_task.status = "Finished"
            self.master.tracker.log_task_end(new_task.job_id, new_task.task_id, self.node_id, start_time, end_time,)

    def process_tasks_overlap(self):

        while True:
            self.new_task = yield self.queue.get()
            
            if not self.new_task: continue
            
            if not self.checkForJobs(self.new_task.job_id) and self.new_task != None:
                yield self.env.timeout(0.01)
                yield self.queue.put(self.new_task)
            elif self.new_task != None:
                self.free_node = False

                self.running_task.append(self.new_task)

                if not self.new_task.job_id in self.datasets:
                    yield self.new_task.dataset_ready_event
                    self.datasets.append(self.new_task.job_id)

                start_time = self.env.now
                self.new_task.status = "Started"
                self.new_task.node= self.node_id
                self.new_task.starting_time = start_time
                self.master.tracker.log_task_start(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time)
                yield self.env.timeout(self.new_task.duration)  # actual processing
                end_time = self.env.now
                self.master.tracker.log_task_end(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time, end_time)

                
                self.free_node = True
                self.running_task.pop(0)
            
    def process_tasks_homogeneous_computes(self):
        
        while True:
            self.new_task = yield self.queue.get()
            #logger.debug("[%s] Compute-%s: Got new task of job %s: duration: %s",
            #             self.env.now, self.node_id, self.new_task.job_id, self.new_task.duration)
            if not self.new_task: continue
            
            if not self.checkForJobs(self.new_task.job_id) and self.new_task != None:
                yield self.env.timeout(0.01)
                yield self.queue.put(self.new_task)

            elif self.new_task != None:
                self.free_node = False

                self.running_task.append(self.new_task)

                if not self.new_task.job_id in self.datasets:
                    yield self.new_task.dataset_ready_event
                    self.datasets.append(self.new_task.job_id)

                start_time = self.env.now
                self.new_task.status = "Started"
                self.new_task.node= self.node_id
                self.new_task.starting_time = start_time
                self.master.tracker.log_task_start(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time)

                task_duration = self.new_task.duration/self.compute_capacity
                yield self.env.timeout(task_duration)  # actual processing
                end_time = self.env.now

                self.master.tracker.log_task_end(self.new_task.job_id, self.new_task.task_id, self.node_id, start_time, end_time)

                self.free_node = True
                self.running_task.pop(0)

    def process_tasks_transfert(self):
        while True:
            new_task = yield self.queue.get()
            
            self.running_task.append(new_task)
            self.running_job = new_task.job_id if self.running_job != new_task.job_id else self.running_job

            if new_task.job_id == -1:
                logger.debug("[%s] Compute-%s: Got new dataset transfert of job %s: duration: %s, dataset_size: %s",
                         self.env.now, self.node_id, new_task.task_id, new_task.duration, new_task.dataset_size)
                
                transfer_time = transferCost(self.master, new_task.dataset_size, self.bandwidth)
                self.free_node = False
                start_time = self.env.now
                yield self.env.timeout(transfer_time)  # actual transfer
                end_time = self.env.now
                self.master.tracker.log_transfer(new_task.task_id, self.node_id,start_time, end_time, new_task.dataset_size, task_id=new_task.task_id)
            else: 
                self.free_node = False
                self.node_ready_event = self.env.event()
                logger.debug("[%s] Compute-%s: Got new task with job_id: %s, task_id: %s, duration: %s, dataset_size: %s",
                            self.env.now, self.node_id, new_task.job_id, new_task.task_id, new_task.duration, new_task.dataset_size)    

                yield new_task.dataset_ready_event
                start_time = self.env.now
                new_task.status = "Started"
                new_task.node= self.node_id
                new_task.starting_time = start_time
                self.master.tracker.log_task_start(new_task.job_id, new_task.task_id, self.node_id, start_time)
                task_duration = new_task.duration*self.compute_capacity
                yield self.env.timeout(task_duration)  # actual processing  # actual processing
                end_time = self.env.now
                self.master.tracker.log_task_end(new_task.job_id, new_task.task_id, self.node_id, start_time, end_time,)
                self.node_ready_event.succeed()
                self.free_node = True
            self.running_task.pop(0)
            self.running_job = None

    def processTasksWithoutParallelisme(self):
        while True:
            new_task = yield self.queue.get()
            
            self.running_task.append(new_task)
            self.running_job = new_task.job_id if self.running_job != new_task.job_id else self.running_job

            if new_task.job_id == -1:
                logger.debug("[%s] Compute-%s: Got new dataset transfert of job %s: duration: %s, dataset_size: %s",
                         self.env.now, self.node_id, new_task.task_id, new_task.duration, new_task.dataset_size)
                
                transfer_time = transferCost(self.master, new_task.dataset_size, self.bandwidth)
                self.free_node = False
                self.running_transfer = True
                start_time = self.env.now
                yield self.env.timeout(new_task.duration)  # actual transfer
                end_time = self.env.now
                self.running_transfer = False
                self.master.tracker.log_transfer(new_task.task_id, self.node_id, end_time - transfer_time, end_time, new_task.dataset_size, task_id=new_task.task_id)
            else: 
                self.free_node = False
                self.node_ready_event = self.env.event()
                logger.debug("[%s] Compute-%s: Got new task with job_id: %s, task_id: %s, duration: %s, dataset_size: %s",
                            self.env.now, self.node_id, new_task.job_id, new_task.task_id, new_task.duration, new_task.dataset_size)    

                yield new_task.dataset_ready_event
                start_time = self.env.now
                new_task.status = "Started"
                new_task.node= self.node_id
                new_task.starting_time = start_time
                self.master.tracker.log_task_start(new_task.job_id, new_task.task_id, self.node_id, start_time)
                yield self.env.timeout(new_task.duration)  # actual processing
                end_time = self.env.now
                self.master.tracker.log_task_end(new_task.job_id, new_task.task_id, self.node_id, start_time, end_time,)
                self.node_ready_event.succeed()
                self.free_node = True
            self.running_task.pop(0)
            self.running_job = None

    def checkForJobs(self, job_id, arriving_time = None):
        if job_id not in self.datasets:
            return False
        else:
            True
        for job in self.master.running_jobs:#.index
            return True
        return True

    def checkOrder(self,job_id):
        if len(self.job_order) == 0 or self.job_order[0] == job_id:
            return False
        return True
    
    def sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time=None, f1=1, f2=1):
        """
            Sort the possible nodes based on a combination of bandwidth and VCPU
        """
        sorted_nodes = []
        reference_time = job.tasks[0].duration
        for node_info in possible_nodes:
            transfer_to_node_j = transferCost(master, dataset_size=job.dataset_size, node_bw=node_info[0].bandwidth)
            time_to_start = transfer_to_node_j
            
            #node_importance = f1 * transfer_to_node_j + max(0, node_info[1] - transfer_to_node_j)
            node_importance = (f1 * transfer_to_node_j) + (f2 * ( reference_time / node_info[0].compute_capacity))

            sorted_nodes.append([node_info[0], time_to_start, node_importance])
        return sorted(sorted_nodes, key=lambda x: x[2], reverse=False)

    def sortOnTransfersTime(master, job, possible_nodes):
        """
            Sort the possible nodes on the transfer time
        """
        sorted_nodes = []
        for node_info in possible_nodes:
            transfer_to_node_j = transferCost(master, dataset_size=job.dataset_size, node_bw=node_info[0].bandwidth)
            time_to_start = transfer_to_node_j + max(0, node_info[1] - transfer_to_node_j)
            sorted_nodes.append([node_info[0], time_to_start])
        
        return sorted(sorted_nodes, key=lambda x:x[1])

    def sortByComputeCapacity(master, job, possible_nodes):
        """
            Sort the possible nodes on the compute capacity
        """
        sorted_nodes = []
        for node_info in possible_nodes:
            transfer_to_node_j = transferCost(master, dataset_size=job.dataset_size, node_bw=node_info[0].bandwidth)
            time_to_start = transfer_to_node_j
            sorted_nodes.append([node_info[0], time_to_start, node_info[0].compute_capacity])
        
        return sorted(sorted_nodes, key=lambda x:x[2], reverse=False)

    def sortByEnergyConsumption(master, job, possible_nodes):
        """
            Sort the possible nodes on the energy consumption
        """
        sorted_nodes = []
        for node_info in possible_nodes:
            transfer_to_node_j = transferCost(master, dataset_size=job.dataset_size, node_bw=node_info[0].bandwidth)
            time_to_start = transfer_to_node_j
            sorted_nodes.append([node_info[0], time_to_start, node_info[0].energy_consumption])
        
        return sorted(sorted_nodes, key=lambda x:x[2], reverse=False)

    def sortingBy(master, job, possible_nodes, criteria, reference_time=None, f1=1, f2=1):
        """
            Sort the possible nodes based on the given criteria
        """
        if len(possible_nodes) == 0: return []
        if criteria == "transfer_time":
            return ComputeNode.sortOnTransfersTime(master,job, possible_nodes)
        elif criteria == "compute_capacity":
            return ComputeNode.sortByComputeCapacity(master, job, possible_nodes)
        elif criteria == "energy_consumption":
            return ComputeNode.sortByEnergyConsumption(master, job, possible_nodes)
        elif criteria == "bandwidth_vcpu":
            return ComputeNode.sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time, f1, f2)

        elif criteria == "bandwidth_vcpu-fixed":
            # (alpha, beta) fixed globally (same for every job), read from the config --
            # used for the grid search (cf. xp_grid_search_weights.py), no per-job_type/
            # learning logic here.
            alpha = master._config.get('fixed_alpha', 1.0)
            beta = master._config.get('fixed_beta', 1.0)
            return ComputeNode.sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time, alpha, beta)

        elif criteria == "bandwidth_vcpu-pools":
            # HARD partition of nodes by learned pool (cf. xp_job_pools_test.py: KMeans
            # over dataset_size/task_duration/nb_tasks -> master.job_classifier, node
            # pools -> master.node_pools). Falls back to the full set if no node from
            # the preferred pool is currently free (avoids starvation).
            pool_id = getattr(job, "learned_pool", None)
            allowed = master.node_pools.get(pool_id, set()) if hasattr(master, "node_pools") else set()
            filtered = [n for n in possible_nodes if n[0].node_id in allowed]
            candidates = filtered if filtered else possible_nodes
            return ComputeNode.sortByBandwidthAndVCPU(master, job, candidates, reference_time, 1.0, 1.0)

        elif criteria == "bandwidth_vcpu-jobtype":
            # Deterministic rule (no learning): (alpha, beta) weights fixed per
            # type_of_job, chosen from the profile observed on inst1-200j-50Nodes
            # (type 0: medium dataset + short tasks -> transfer dominates -> high alpha;
            #  type 1: tiny dataset + long tasks -> compute dominates -> high beta;
            #  type 2: large dataset + long tasks -> both matter -> balanced).
            job_type_weights = {0: (2.0, 0.3), 1: (0.3, 2.0), 2: (1.0, 1.0)}
            alpha, beta = job_type_weights.get(getattr(job, "type_of_job", None), (1.0, 1.0))
            return ComputeNode.sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time, alpha, beta)

        elif criteria == "bandwidth_vcpu-adaptative":
            #return ComputeNode.sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time, 1, 1)
            min_bandwidth = np.min([node[0].bandwidth for node in possible_nodes])
            min_cpu = np.min([node[0].compute_capacity for node in possible_nodes])
            
            #if job.params == None:
            (alpha, beta) = estimatingParams(master, job, len(possible_nodes),min_bandwidth,min_cpu)
            job.params = (alpha,beta)
            #else:
            #    alpha, beta = job.params
            #print("Parametre choisis = ", (alpha, beta))
            if master._config['checkAvailableNodes']:
                nodes_ = ComputeNode.checkIFAccepted(alpha, beta, master, possible_nodes)
                return ComputeNode.sortByBandwidthAndVCPU(master, job, nodes_, reference_time, alpha, beta)
            else:
                return ComputeNode.sortByBandwidthAndVCPU(master, job, possible_nodes, reference_time, alpha, beta)
        else:
            raise ValueError(f"Unknown sorting criteria: {criteria}")
    
    def sortingByAdaptative(master, possible_nodes):
        """
            Sort according to the better
        """
        max_bandwidth = np.max([node.bandwidth for node in master.compute_nodes])
        max_computer_capacity = np.max([1/node.compute_capacity for node in master.compute_nodes])
        nodes_importances = []
        for node_info in possible_nodes:
            node_importance = (max_bandwidth / node_info[0].bandwidth ) + ((1/ node_info[0].compute_capacity) / max_computer_capacity  )
            nodes_importances.append((node_info[0], node_importance))

        return sorted(nodes_importances, key=lambda x: x[1], reverse=False)

    def sortAllNodes(master):
        max_bandwidth = np.max([node.bandwidth for node in master.compute_nodes])
        max_computer_capacity = np.max([1/node.compute_capacity for node in master.compute_nodes])
        nodes_importances = []
        for node_info in master.compute_nodes:
            node_importance = (max_bandwidth / node_info.bandwidth ) + ((1/ node_info.compute_capacity) / max_computer_capacity  )
            nodes_importances.append((node_info, node_importance))

        return sorted(nodes_importances, key=lambda x: x[1], reverse=False)

    def checkIFAccepted(alpha, beta,master_nodes, possible_nodes):
        #avg_bandwidth_dispo = np.min([node[0].bandwidth for node in possible_nodes])
        #max_compute_dispo = np.max([node[0].compute_capacity for node in possible_nodes])
        #min_compute_dispo = np.min([node[0].compute_capacity for node in possible_nodes])
        #avg_vcpu_dispo = np.mean([node[0].compute_capacity for node in possible_nodes])
        #return True
        """if avg_bandwidth_dispo >= master_nodes.avg_bandwidth and avg_vcpu_dispo <= master_nodes.avg_vcpu:
            return [node[0].bandwidth for node in possible_nodes]
        else:
            return [node[0].bandwidth for node in possible_nodes]
        if alpha == beta:
            return possible_nodes
        else:"""
        if alpha > beta: 
            candidates = [node for node in possible_nodes if node[0].bandwidth >= (master_nodes.avg_bandwidth)]        
            return candidates if len(candidates) > 0 else []
        elif alpha < beta:
            candidates = [node for node in possible_nodes if node[0].compute_capacity <= (master_nodes.avg_compute_capacity)]
            return candidates if len(candidates) > 0 else []
        else:
            candidates = [node for node in possible_nodes if node[0].bandwidth >= (master_nodes.avg_bandwidth) and node[0].compute_capacity <= (master_nodes.avg_compute_capacity)]
            return candidates if len(candidates) > 0 else []
                