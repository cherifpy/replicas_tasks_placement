import math
import random
import simpy
import numpy as np
import logging
from job import Task, Replica, Job
import copy
from compute_node import ComputeNode
from utils.searchAlgo import sortReplicasByNodesBw, sortReplicasByNodesVCPU, estimateFlowTime
import heuristic_scheduler

logger = logging.getLogger(__name__)

def transferCost(self,dataset_size, node_bw = None, config = None):
        bw = node_bw if node_bw else self._config['compute_node_bw_MBps']
        ls = self._config['compute_node_latency_ms']
        return dataset_size/bw

def changingPriority(running_jobs, ascending=True):

    sorted_jobs = []
    for job in running_jobs:
        
        if job.task_execution_time == None:
            sorted_jobs.append((job.job_id, float("inf")))  
        else:
            sorted_jobs.append((job.job_id, job.task_execution_time*job.nb_tasks))
    return sorted(sorted_jobs, key=lambda x: x[1], reverse= not ascending)

from dataclasses import dataclass, field

@dataclass
class UtilityCheckResult:
    nb_acceptable_nodes: int
    nodes_and_replicas: dict
    acceptable_nodes: list
    makspan_with_acceptable_nodes: float = None
    makespan_per_replica: dict = None
    # node_id -> {'action': 'insert'}, see heuristic_scheduler.
    node_actions: dict = field(default_factory=dict)

class UtilityBasedHeterogeneousApproach:
    """Master node handles job submissions."""
    def __init__(self, env, compute_nodes, tracker, config, overlap=False):
        self.env = env
        self.queue = simpy.Store(env)
        self.compute_nodes:list[ComputeNode] = compute_nodes
        self.tracker = tracker
        self._config = config
        self.all_jobs = {}
        self.running_jobs:list[Job] = []
        self.waiting_jobs = []
        self.finished_jobs = 0
        self.replicas_stats = {}
        self.replicas_placements = {}
        self.nb_nodes = len(compute_nodes)
        self.actual_transfers = {}
        self.x = 1
        self.overlap = self._config['overlap'] if 'overlap' in self._config else overlap
        self.replicas_locations = {}
        self.threshold = config['threshold']
        self.all_flow_times = []
        self.avg_bandwidth = np.mean([node.bandwidth for node in self.compute_nodes])
        self.avg_compute_capacity = np.mean([node.compute_capacity for node in self.compute_nodes])
        # Explicit topology matrix - None if config['topology_matrix_path'] isn't
        # configured/the file doesn't exist, in which case topologyCostFactor falls back
        # to the old synthetic ring based on node_id (heuristic_scheduler.py).
        self.topology_matrix = heuristic_scheduler.load_topology_matrix(config.get('topology_matrix_path'))
        self.estimated_flow_times = {}
        self.feature_dim = 6
        self.dataset_sizes = []
        self.print =True
        if config['rl-agent']:
            from utils.RLAgent import LinUCBAgent,DiscreteReinforceAgent
            self.rl_agent = LinUCBAgent(feature_dim=self.feature_dim)
        else:
            self.rl_agent = None

        logging.debug(f"Master node with {self.nb_nodes} compute nodes")

    def sendJobOnOneNode(self, heterogeneous=False):

        """
            Migration must also trigger here if the cost is high
        """
        self.job_priorities = {node.node_id:[] for node in self.compute_nodes}
        while True:
            logger.debug("[%s] Master: Waiting for new job", self.env.now)
            new_job = yield self.queue.get()
            job_started = False
            new_job.arriving_time = self.env.now
            self.all_jobs[new_job.job_id] = {'dataset_size': new_job.dataset_size, 'nb_rescheduled': 0}
            self.tracker.register_job(new_job.job_id, self.env.now)
            
           
            free_compute_nodes = self.node_to_use_to_starts(self._config['total_nb_compute_nodes'],new_job, heterogeneous=heterogeneous, first=True)#self.select_free_nodes(k=-1)##self.select_free_nodes(k=k)

            if len(free_compute_nodes) == 0 or len(self.waiting_jobs) > 0:
                logger.debug("[%s] Master: Not enough free nodes available for job %s. Rescheduling.", self.env.now, new_job.job_id)
                self.waiting_jobs.append(new_job)

            else:
                logger.debug("Job %s submitted From 1st try", new_job.job_id)  
                
                free_compute_nodes = self.node_to_use_to_starts(self._config['total_nb_compute_nodes'],new_job, heterogeneous=heterogeneous, first=True)#self.select_free_nodes(k=k)
                
                if len(free_compute_nodes) >= 1:
                    compute_node = free_compute_nodes[0][0]
                    new_job.starting_time = self.env.now
                    self.dataset_sizes.append(new_job.dataset_size)

                    if new_job.nb_remaining_tasks == 0: break
                    task = new_job.tasks[0] 
                    dataset_ready_event = self.env.event()

                    if compute_node.node_id not in new_job.replicas_nodes:
                        new_job.replicas_nodes.append(compute_node.node_id)
                        new_job.transfer_time =  transferCost(self,new_job.dataset_size,compute_node.bandwidth) # new_job.dataset_size / 
                        replica_inst =  Replica(new_job.job_id, node_id=compute_node.node_id, data_size=new_job.dataset_size, transfer_time=new_job.transfer_time,makespan=float('inf'),transfer_start_time=self.env.now)
                        self.replicas_stats[(new_job.job_id, compute_node.node_id)] = replica_inst
                        new_job.replicas.append(replica_inst)
                        
                        dataset_ready_event = self.env.event()

                        self.env.process(self.transfer_data(new_job.job_id, new_job.dataset_size, compute_node,dataset_ready_event))
                        
                        new_job.nb_replicas +=1
                        new_job.node_referent = compute_node.node_id
                        new_job.actual_makespan = self.env.now + new_job.transfer_time + (len(new_job.tasks)*task.duration/compute_node.compute_capacity)
                        compute_node.current_dataset = new_job.job_id
                    
                    task.dataset_ready_event = dataset_ready_event
                    new_job.nb_remaining_tasks -= 1
                    task.node = compute_node.node_id
                    self.replicas_stats[(new_job.job_id, task.node)].nb_tasks +=1
                    self.replicas_stats[(new_job.job_id, task.node)].task_execution_time += task.duration/compute_node.compute_capacity
                    task.status = "Scheduled"

                    yield compute_node.queue.put(task)
                    logger.debug("[%s] Master 1: Sent task %s from job %s to node %s", self.env.now, task.task_id,new_job.job_id,task.node)
                    job_started = True

                else:
                    #check if there is not another alternative node to use for this job, if yes, send the job to this node
                    pass
                   
                if job_started:
                    job_started = False
                    self.running_jobs.append(new_job)
                else:
                    logger.debug("[%s] Master: Not enough free nodes available for job %s. Rescheduling.", self.env.now, new_job.job_id)
                    self.waiting_jobs.append(new_job)
                
            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.all_jobs) == self._config['total_nb_jobs'] and len(self.tracker.ongoing_tasks) == 0 and len(self.queue.items) == 0:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break

    def _placeNewJobFirstReplica(self, new_job, compute_node):

        new_job.starting_time = self.env.now
        task = new_job.tasks[0]
        dataset_ready_event = self.env.event()

        if compute_node.node_id not in new_job.replicas_nodes:
            new_job.replicas_nodes.append(compute_node.node_id)
            new_job.transfer_time = transferCost(self, new_job.dataset_size, compute_node.bandwidth)
            replica_inst = Replica(new_job.job_id, node_id=compute_node.node_id, data_size=new_job.dataset_size,
                                   transfer_time=new_job.transfer_time, makespan=float('inf'), transfer_start_time=self.env.now)
            new_job.replicas.append(replica_inst)
            self.replicas_stats[(new_job.job_id, compute_node.node_id)] = replica_inst

            dataset_ready_event = self.env.event()
            self.env.process(self.transfer_data(new_job.job_id, new_job.dataset_size, compute_node,
                                        dataset_ready_event))

            new_job.nb_replicas +=1
            new_job.node_referent = compute_node.node_id
            new_job.actual_makespan = self.env.now + new_job.transfer_time + (len(new_job.tasks)*task.duration/compute_node.compute_capacity)
            compute_node.current_dataset = new_job.job_id

        task.dataset_ready_event = dataset_ready_event
        new_job.nb_remaining_tasks -= 1
        task.node = compute_node.node_id
        self.replicas_stats[(new_job.job_id, task.node)].nb_tasks +=1
        self.replicas_stats[(new_job.job_id, task.node)].task_execution_time += task.duration/compute_node.compute_capacity
        task.status = "Scheduled"

        yield compute_node.queue.put(task)
        logger.debug("[%s] Master 2: Sent task %s from job %s to node %s", self.env.now, task.task_id,new_job.job_id,task.node)

    def sendNoStartedJobsOrTasks(self, heterogeneous=False):
        to_delete = []
        while True:
            yield self.env.timeout(0.15)
            if len(self.waiting_jobs) > 0:

                new_job = self.waiting_jobs[0]
                free_compute_nodes = self.node_to_use_to_starts(self._config['total_nb_compute_nodes'],new_job, heterogeneous=heterogeneous, first=True)#self.select_free_nodes(k=k)
                best_free_node = free_compute_nodes[0][0] if free_compute_nodes else None

                job_started = False
                if best_free_node is not None:
                    self.dataset_sizes.append(new_job.dataset_size)
                    yield from self._placeNewJobFirstReplica(new_job, best_free_node)
                    job_started = True

                if job_started:
                    _ = self.waiting_jobs.pop(0)
                    self.running_jobs.append(new_job)

            
            for i,job in enumerate(self.running_jobs):
                
                for task in job.tasks:
                    if task.status == "Started" and task.starting_time + (task.duration/self.compute_nodes[task.node].compute_capacity) <= self.env.now and task.node != job.node_referent:
                        task.status = "Finished"
                        job.task_execution_time = task.duration
                        
                        not_executed_tasks = job.getNoTExecutedTasks()
                        
                        if  task.node in self.actual_transfers.keys() and self.actual_transfers[task.node][1] > (self.env.now + task.duration/self.compute_nodes[task.node].compute_capacity) and task.node in job.replicas_nodes:
                            job.replicas_nodes.remove(task.node)
                            continue

                        if task.node in job.replicas_nodes and len(not_executed_tasks) != 0:
                            new_task = not_executed_tasks[0]
                            dataset_ready_event = self.env.event()
                            
                            self.replicas_stats[(job.job_id, task.node)].nb_tasks +=1
                            
                            new_task.node = task.node
                            new_task.status = "Scheduled"
                            yield self.compute_nodes[task.node].queue.put(new_task)

                            new_task.dataset_ready_event = dataset_ready_event.succeed()
                            logger.debug("[%s] Master: Sent task %s from job %s to node %s", self.env.now, task.task_id,job.job_id,new_task.node)
                            job.nb_remaining_tasks -= 1


                        elif task.node in job.replicas_nodes:
                            job.replicas_nodes.remove(task.node)
                            if job.job_id in self.compute_nodes[task.node].running_job:
                                self.compute_nodes[task.node].running_job.remove(job.job_id)
                            if job.job_id in self.compute_nodes[task.node].job_order:
                                self.compute_nodes[task.node].job_order.remove(job.job_id)
                            if self.compute_nodes[task.node].current_dataset == job.job_id:
                                heuristic_scheduler.LibererNoeud(self.compute_nodes[task.node])
                                


                if len([task for task in job.tasks if task.status == "Finished"]) == len(job.tasks):
                    job.finish_time = self.env.now
                    to_delete.append((i,job.job_id)) 

            for j,job_id in to_delete:
                for l,job in enumerate(self.running_jobs):
                    if self.running_jobs[l].job_id == job_id:
                        _ = self.running_jobs.pop(l)
                        self.tracker.log_end_job(job.job_id,len(job.tasks),job.dataset_size,job.arriving_time,job.starting_time, job.finish_time,job.transfer_time, job.tasks[0].duration, job.nb_replicas, job.first_optimal_replica_number,job.nb_first_replicas_sended)
                        self.all_flow_times.append(job.finish_time - job.arriving_time)
                        if self.rl_agent:
                            reward = -(job.finish_time - job.arriving_time)  # the shorter the job, the better the reward
                            self.rl_agent.update(job.job_id, reward)
                        self.finished_jobs += 1
                        break
            
            to_delete =[]

            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.all_jobs) == self._config['total_nb_jobs'] and len(self.tracker.ongoing_tasks) == 0 and len(self.queue.items) == 0:
                finished = True
                for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break
    
    def reschedulTasks(self, condition_only=False, heterogeneous = False):
        to_delete =[]
        theta = self._config['threshold']
        while True:
            yield self.env.timeout(0.1)

            jobs_sorted = []
            if self._config['changing_priority']:
                jobs_sorted = changingPriority(self.running_jobs, ascending=self._config['ascending'])
            else:
                jobs_sorted = [(job.job_id,0) for job in self.running_jobs]

            for i,job_info in enumerate(jobs_sorted):
                job = None
                for j in self.running_jobs:
                    if j.job_id == job_info[0]:
                        job = j
                        break
                if job == None: continue

                if len(job.replicas_nodes) == 0 and job.nb_remaining_tasks > 0:
                    
                    not_executed = job.getNoTExecutedTasks()
                    if not_executed:
                        free_compute_nodes = self.node_to_use_to_starts(
                            self._config['total_nb_compute_nodes'], job, heterogeneous=heterogeneous, first=True
                        )
                        if len(free_compute_nodes) > 0:
                            compute_node = free_compute_nodes[0][0]
                            task = not_executed[0]
                            dataset_ready_event = self.env.event()

                            job.replicas_nodes.append(compute_node.node_id)
                            job.transfer_time = transferCost(self, job.dataset_size, compute_node.bandwidth)
                            replica_inst = Replica(
                                job.job_id, node_id=compute_node.node_id, data_size=job.dataset_size,
                                transfer_time=job.transfer_time, makespan=float('inf'), transfer_start_time=self.env.now
                            )
                            job.replicas.append(replica_inst)
                            self.replicas_stats[(job.job_id, compute_node.node_id)] = replica_inst

                            self.env.process(self.transfer_data(job.job_id, job.dataset_size, compute_node, dataset_ready_event))

                            job.nb_replicas += 1
                            job.node_referent = compute_node.node_id
                            job.actual_makespan = self.env.now + job.transfer_time + (job.nb_remaining_tasks * task.duration / compute_node.compute_capacity)
                            compute_node.current_dataset = job.job_id

                            task.dataset_ready_event = dataset_ready_event
                            job.nb_remaining_tasks -= 1
                            task.node = compute_node.node_id
                            self.replicas_stats[(job.job_id, task.node)].nb_tasks += 1
                            self.replicas_stats[(job.job_id, task.node)].task_execution_time += task.duration / compute_node.compute_capacity
                            task.status = "Scheduled"

                            yield compute_node.queue.put(task)
                            logger.debug("[%s] Master: Job %s re-seeded on node %s after losing all its replicas.",
                                         self.env.now, job.job_id, compute_node.node_id)
                    continue

                if job.nb_remaining_tasks > 0 and not any(t.status in ("Started", "Scheduled") for t in job.tasks):
                    
                    not_executed = job.getNoTExecutedTasks()
                    if not_executed:
                        target_node_id = job.node_referent if job.node_referent in job.replicas_nodes else job.replicas_nodes[0]
                        target_node = self.compute_nodes[target_node_id]
                        new_task = not_executed[0]
                        self.replicas_stats[(job.job_id, target_node_id)].nb_tasks += 1
                        new_task.node = target_node_id
                        new_task.status = "Scheduled"
                        dataset_ready_event = self.env.event()
                        yield target_node.queue.put(new_task)
                        new_task.dataset_ready_event = dataset_ready_event.succeed()
                        job.nb_remaining_tasks -= 1
                        logger.debug("[%s] Master: Job %s - task %s re-attached to its existing node %s (orphaned).",
                                     self.env.now, job.job_id, new_task.task_id, target_node_id)
                    continue

                case, bol, t_unit, t_reference =  self.condition(job)
                if bol:
                    task = [task for task in job.tasks if task.node == job.node_referent and task.status == "Started"][0]

                    if case == 1:
                        task.status = "Finished"
                        job.task_execution_time = task.duration
                        
                        not_executed_tasks = job.getNoTExecutedTasks()
                        if len(not_executed_tasks) != 0 and task.node in job.replicas_nodes:
                            new_task = not_executed_tasks[0]
                            self.replicas_stats[(job.job_id, task.node)].nb_tasks +=1
                            new_task.node = task.node
                            new_task.status = "Scheduled"
                            dataset_ready_event = self.env.event()
                            yield self.compute_nodes[task.node].queue.put(new_task)
                            new_task.dataset_ready_event = dataset_ready_event.succeed()  # Inject the ready event
                            logger.debug("[%s] Master: Sent task %s from job %s to node %s", self.env.now, task.task_id,job.job_id,new_task.node)
                            job.nb_remaining_tasks -= 1
                        elif task.node in job.replicas_nodes:
                            job.replicas_nodes.remove(task.node)
                            if job.job_id in self.compute_nodes[task.node].running_job:
                                self.compute_nodes[task.node].running_job.remove(job.job_id)
                            if job.job_id in self.compute_nodes[task.node].job_order:
                                self.compute_nodes[task.node].job_order.remove(job.job_id)
                            if self.compute_nodes[task.node].current_dataset == job.job_id:
                                heuristic_scheduler.LibererNoeud(self.compute_nodes[task.node])


                free_node = self.node_to_use_to_starts(self._config['total_nb_compute_nodes'],job, heterogeneous=heterogeneous, reference_time=t_reference)#self.select_free_nodes(k=-1)
                
                if t_reference != None and len(job.getNoTExecutedTasks()) > 0 and (job.nb_tasks - job.nb_replicas) > 0 and len(free_node) > 0: #max_nb_replicas != None and 
                    
                    #max_replicas_to_add = len(free_node) if len(free_node) < (max_nb_replicas - job.nb_replicas) else int(max_nb_replicas - job.nb_replicas)
                    node_to_use = copy.copy(free_node)
                    
                    """if theta != self._config['threshold']:
                        print('thetaaaa chaanggeeeeed', theta, self._config['threshold'])"""
                    

                    data  =  self.checkUtilityProblem(job, node_to_use)
                    
                    nb_replicas_to_add = data.nb_acceptable_nodes
                    nodes_and_replicas = data.nodes_and_replicas
                    nodes_for_replication = data.acceptable_nodes
                    # {data.acceptable_nodes}. Makespan went from {job.actual_makespan} to {data.makspan_with_acceptable_nodes}")
                    job.actual_makespan = data.makspan_with_acceptable_nodes
                    makespan_peer_replicas = data.makespan_per_replica
                    
                    
                
                    not_executed_tasks = job.getNoTExecutedTasks()

                    if nb_replicas_to_add == 0: continue
                    
                    for replica in job.replicas:
                            replica.makespan = makespan_peer_replicas[replica.node_id]

                    node_actions = data.node_actions

                    for i in range(nb_replicas_to_add):
                        if i >= len(not_executed_tasks): break
                        compute_node = self.compute_nodes[nodes_for_replication[i]]
                        new_task = not_executed_tasks.pop(0)
                        if compute_node.node_id not in job.replicas_nodes:
                            action_info = node_actions.get(compute_node.node_id, {'action': 'insert'})
                            dataset_ready_event = heuristic_scheduler.AppliquerSelection(
                                self, job, compute_node, action_info, makespan_peer_replicas
                            )
                            if dataset_ready_event is None:
                                # The selected action (migration/eviction) is no longer safe:
                                # the task stays NotStarted, to be picked up again later.
                                continue
                        else:
                            dataset_ready_event = self.env.event()

                        new_task.dataset_ready_event = dataset_ready_event
                        job.nb_remaining_tasks -= 1
                        new_task.node = compute_node.node_id
                        self.replicas_stats[(job.job_id, new_task.node)].nb_tasks +=1
                        self.replicas_stats[(job.job_id, new_task.node)].task_execution_time += new_task.duration/compute_node.compute_capacity
                        new_task.status = "Scheduled"
                        yield compute_node.queue.put(new_task)
                        logger.debug("[%s] Master: Sent task %s from job %s to node %s", self.env.now, new_task.task_id,job.job_id,new_task.node)

                if len([task for task in job.tasks if task.status == "Finished"]) == len(job.tasks):
                    job.finish_time = self.env.now
                    to_delete.append((i,job.job_id)) 


                for j,job_id in to_delete:
                    for i,job in enumerate(self.running_jobs):
                        if self.running_jobs[i].job_id == job_id:
                            job = self.running_jobs.pop(i)
                            self.tracker.log_end_job(job.job_id,len(job.tasks),job.dataset_size,job.arriving_time,job.starting_time, job.finish_time,job.transfer_time, job.tasks[0].duration, job.nb_replicas, job.first_optimal_replica_number,job.nb_first_replicas_sended)
                            self.all_flow_times.append(job.finish_time - job.arriving_time)
                            if self.rl_agent:
                                reward = -(job.finish_time - job.arriving_time)  # the shorter the job, the better the reward
                                self.rl_agent.update(job.job_id, reward)
                            self.finished_jobs += 1
                            break
                to_delete =[]

            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.all_jobs) == self._config['total_nb_jobs'] and len(self.tracker.ongoing_tasks) == 0 and len(self.queue.items) == 0:
                finished = True
                for compute_node in self.compute_nodes:
                    if len(compute_node.queue.items) > 0:
                        finished = False
                if finished:
                    break

    def checkUtilityProblemV1(self, job:Job,free_nodes,  condition_only=False):

        replicas_to_add = []
        acceptable_node = []
        all_excuted_task = 0
        old_replicas_and_nodes = {}

        t_now = self.env.now
        t_reference = job.tasks[0].duration
        for node_info in range(len(free_nodes)):
            
            transfer_time = transferCost(self,job.dataset_size, free_nodes[node_info][0].bandwidth)
            new_replicas = Replica(job.job_id, node_id=free_nodes[node_info][0].node_id, data_size=job.dataset_size,makespan=0,transfer_time=transfer_time, transfer_start_time=t_now)

            all_excuted_task = 0
            replicas_to_check = sortReplicasByNodesBw(job.replicas + replicas_to_add + [new_replicas], self)

            nodes_and_replicas = {}
            a_rep = replicas_to_check[-1]
            a_rep_transfer_time = transferCost(self, job.dataset_size, self.compute_nodes[a_rep.node_id].bandwidth)
            nodes_and_replicas[a_rep.node_id] = 0

            for i,rep in enumerate(replicas_to_check[0:-1]):
                rep_transfer_time = transferCost(self, job.dataset_size, self.compute_nodes[rep.node_id].bandwidth)

                nb_task = (((a_rep.transfer_start_time + a_rep_transfer_time) - (rep.transfer_start_time + rep_transfer_time))/(t_reference / self.compute_nodes[rep.node_id].compute_capacity))+1

                if 1 > nb_task > 0: nb_task = 0
                else: nb_task  = int(nb_task)

                all_excuted_task = all_excuted_task + nb_task
                nodes_and_replicas[rep.node_id] = nb_task

                if all_excuted_task > job.nb_remaining_tasks:
                    #if job.job_id == 4: print("S1: Acceptable nodes for job", job.job_id, ":", acceptable_node)
                    
                    if len(acceptable_node)>0:acceptable_node.pop(-1)
                    
                    return len(acceptable_node), True, old_replicas_and_nodes, acceptable_node if len(acceptable_node)>0 else False 

            remaining_tasks = job.nb_remaining_tasks - all_excuted_task
            
            if remaining_tasks <= 0:
                #if job.job_id == 4: print("S2: Acceptable nodes for job", job.job_id, ":", acceptable_node)
                if len(acceptable_node)>0:acceptable_node.pop(-1)
                return len(acceptable_node), True, old_replicas_and_nodes, acceptable_node if len(acceptable_node)>0 else False 
            
            #replicas_to_check = sortReplicasByNodesVCPU(job.replicas + replicas_to_add+ [new_replicas], self)
            max_makespan, nb_task_per_nodes = self.alloc_branch_and_bound(remaining_tasks, replicas_to_check)
            
            add = True
            
            for i,nb_task_per_node in enumerate(nb_task_per_nodes):
                
                replica = replicas_to_check[i]
                
                replica_transfer_time = transferCost(self, job.dataset_size, self.compute_nodes[replica.node_id].bandwidth)
                
                nodes_and_replicas[replica.node_id] += nb_task_per_node
                
                if nb_task_per_node+nodes_and_replicas[replica.node_id] > 0 and replica_transfer_time / ((nb_task_per_node+nodes_and_replicas[replica.node_id]) * t_reference / self.compute_nodes[replica.node_id].compute_capacity) > 1:
                    add = False
                    break
                elif nb_task_per_node+nodes_and_replicas[replica.node_id] < 0:
                    add = False
                    break 

            if add:
                acceptable_node.append(free_nodes[node_info][0].node_id)
                #if job.job_id == 4: print("S2: Acceptable nodes for job", job.job_id, ":", acceptable_node)
            else:
                #if job.job_id == 4: print("S3: Acceptable nodes for job", job.job_id, ":", acceptable_node)
                if len(acceptable_node)>0:acceptable_node.pop(-1)
                return len(acceptable_node), True, old_replicas_and_nodes, acceptable_node if len(acceptable_node)>0 else False  
                return len(acceptable_node), True, old_replicas_and_nodes if len(acceptable_node)>0 else False

            old_replicas_and_nodes = copy.copy(nodes_and_replicas)
        #if job.job_id == 4: print("S4: Acceptable nodes for job", job.job_id, ":", acceptable_node)
        if len(acceptable_node)>0:acceptable_node.pop(-1)
        return len(acceptable_node), True, nodes_and_replicas, acceptable_node if len(acceptable_node)>0 else False       

    def _compute_pre_executable_tasks(self, replicas_to_check, job) -> tuple[dict, int]:
        """
        Computes the tasks each node can execute before the slowest node has
        finished its transfer.

        Returns (nodes_and_replicas, total_executed_tasks)
        """
        nodes_and_replicas = {}
        total_executed = 0
        t_reference = job.tasks[0].duration
        replicas_to_check = sortReplicasByNodesBw(replicas_to_check, self)
        slowest = replicas_to_check[-1]
        slowest_transfer = transferCost(self, job.dataset_size, self.compute_nodes[slowest.node_id].bandwidth)
        nodes_and_replicas[slowest.node_id] = 0

        for rep in replicas_to_check[:-1]:
            rep_transfer = transferCost(self, job.dataset_size, self.compute_nodes[rep.node_id].bandwidth)
            capacity = self.compute_nodes[rep.node_id].compute_capacity

            nb_task = (
                (slowest.transfer_start_time + slowest_transfer)
                - (rep.transfer_start_time + rep_transfer)
            ) / (t_reference / capacity)

            #if int(nb_task) != nb_task:

            nb_task = max(0, math.ceil(nb_task)) if nb_task >= 1 else 1
            total_executed += nb_task
            nodes_and_replicas[rep.node_id] = nb_task


        return nodes_and_replicas, total_executed

    def _is_node_useful(self, replicas_to_check,nb_tasks_per_replica, job, old_makespan ,new_makespan) -> bool:
        
        """
        Checks that each node does enough work to justify the cost of its data
        transfer.
        """
        max_bandwidth = min([node.bandwidth for node in self.compute_nodes])
        max_compute_capacity = min([node.compute_capacity for node in self.compute_nodes])
        #print("Checking utility for node", replicas_to_check[-1].node_id)
        t_ref = job.tasks[0].duration
        t_reference = job.tasks[0].duration

        for i,replica in enumerate(replicas_to_check):
            
            transfer_time = transferCost(self, job.dataset_size, self.compute_nodes[replica.node_id].bandwidth)
            total_tasks = nb_tasks_per_replica[replica.node_id]
            #if job.job_id == 4: 
            #    print(f"Node {replica.node_id} - Transfer time: {transfer_time:.2f}s, Total tasks: {total_tasks}")
            if total_tasks <= 0:
                return False

            utility_ratio = transfer_time / (
                total_tasks * t_ref / self.compute_nodes[replica.node_id].compute_capacity
            )
            #bw_i = self.compute_nodes[replica.node_id].bandwidth
            #bw_ref = 363 #.50

            #cpu_i = self.compute_nodes[replica.node_id].compute_capacity
            #cpu_ref = 4.40
            
            
            improvement = old_makespan / (old_makespan - new_makespan)   if old_makespan-new_makespan > 0 else 0
            threshold = 0.01  # e.g. 5% minimum improvement
            if improvement <= threshold:
                accel = 0
            else:
                accel = 1 + (improvement - threshold) / (1 - threshold)
 
            #accel =  1 if old_makespan - new_makespan > 0 else 0 #old_makespan / (old_makespan -  new_makespan)
            #if accel == 0: print("Acceleration factor:", accel, "Improvement:", improvement)

            #Marche bien
            #accel = old_makespan / (old_makespan - new_makespan)   if old_makespan>new_makespan else 0 #(bw_i / bw_ref ) + (1 / (cpu_i / cpu_ref) )
            #if (old_makespan - new_makespan) <=0:
            #    return False
            #utility_ratio = transfer_time / (
            #    (old_makespan - new_makespan) #total_tasks * t_reference / self.compute_nodes[replica.node_id].compute_capacity
            #)
            #if job.job_id ==0:
            #    pass
                #print(f"Node {replica.node_id} - gainded {(old_makespan - new_makespan)}")# Transfer time: {transfer_time:.2f}s, Total tasks: {total_tasks}, Old makespan: {old_makespan:.2f}s, New makespan: {new_makespan:.2f}s, Utility ratio: {utility_ratio:.2f}, Acceleration factor: {accel:.2f}")
            #print("Valeur de accel", accel)
            if utility_ratio > 1 * accel:
                #print(f"Node {replica.node_id} is not useful  transfer {transfer_time} nb task {total_tasks} t ref {t_reference}  cc {self.compute_nodes[replica.node_id].compute_capacity}- Utility ")
                return False

        return True

    def checkUtilityProblem(self, job: Job, free_nodes) -> UtilityCheckResult:

        return heuristic_scheduler.SelectionnerMeilleurNoeud(self, job, free_nodes)
    
    def allocation_of_tasks(self, job: Job, replicas_to_check) -> dict:
        

        ready_at = {}

        nb_tasks = job.nb_tasks
        nb_remaining_tasks = job.nb_tasks
        
        t_reference = job.tasks[0].duration
        
        affected_tasks = {replica.node_id: 0 for replica in replicas_to_check}

        for replica in replicas_to_check:
            cn = self.compute_nodes[replica.node_id]
            t_end = replica.transfer_start_time + transferCost(self, job.dataset_size, cn.bandwidth)
            ready_at[replica.node_id] = t_end - self.env.now
        
        ready_at = dict(sorted(ready_at.items(), key=lambda item: item[1]))
        negatif_ready_nodes = [node_id for node_id, t in ready_at.items() if t < 0]
        i = 0
        while len(negatif_ready_nodes) > 0:
            cn = self.compute_nodes[negatif_ready_nodes[i]]
            ready_at[negatif_ready_nodes[i]] += (t_reference / cn.compute_capacity)
            affected_tasks[negatif_ready_nodes[i]] += 1
            ready_at = dict(sorted(ready_at.items(), key=lambda item: item[1]))
            negatif_ready_nodes = [node_id for node_id, t in ready_at.items() if t < 0]
            nb_remaining_tasks -= 1

        ready_at = dict(sorted(ready_at.items(), key=lambda item: item[1]))

        while nb_remaining_tasks > 0:
            node_id = list(ready_at.keys())[0]
            if nb_remaining_tasks == 0: break
            ready_at[node_id] += (t_reference / self.compute_nodes[node_id].compute_capacity)
            affected_tasks[node_id] += 1
            nb_remaining_tasks -= 1
            ready_at = dict(sorted(ready_at.items(), key=lambda item: item[1]))
        
        ##Estimate makespan total
        makespan_peer_replicas = {}
        for replica in replicas_to_check:
            waiting_time = replica.transfer_start_time + transferCost(self, job.dataset_size, self.compute_nodes[replica.node_id].bandwidth)
            exec_time = affected_tasks[replica.node_id] * (t_reference / self.compute_nodes[replica.node_id].compute_capacity)
            makespan_peer_replicas[replica.node_id] = waiting_time + exec_time 
             
            #print(f"Replica on node {replica.node_id} with transfer time {replica.transfer_time:.2f} and affected tasks {affected_tasks.get(replica.node_id, 0)} makespan in this node estimated to: {makespan_peer_replicas[replica.node_id]:.2f}")

        return max(makespan_peer_replicas.values()), affected_tasks, makespan_peer_replicas                

    def allocate_tasks(self, N, replicas, job):
        """
        N        : number of remaining tasks to distribute
        replicas : list of Replica (existing + candidates)
        job      : the Job in question
        """
        import math
        t_now = self.env.now
        t_ref = job.tasks[0].duration

        nodes = []
        for rep in replicas:
            cn = self.compute_nodes[rep.node_id]
            speed = cn.compute_capacity / t_ref  # tasks per unit of time

            # --- Task currently running on this node? ---
            running_tasks = [
                task for task in job.tasks
                if task.status == "Started" and task.node == rep.node_id
            ]

            if running_tasks:
                # The node is busy -> ready_at = remaining time before the task ends
                task = running_tasks[0]
                task_end = task.starting_time + (task.duration / cn.compute_capacity)
                ready_at = max(0.0, task_end - t_now)

            else:
                # No task running -> ready_at = remaining transfer time
                tr_time  = transferCost(self, job.dataset_size, cn.bandwidth)
                tr_end   = rep.transfer_start_time + tr_time
                ready_at = max(0.0, tr_end - t_now)

            nodes.append({
                'node_id'  : rep.node_id,
                'ready_at' : ready_at,
                'speed'    : speed,
            })

        # --- Bisection over the optimal makespan T* ---
        lo = 0.0
        hi = max(n['ready_at'] for n in nodes) + N * max(1.0 / n['speed'] for n in nodes)

        for _ in range(60):
            mid   = (lo + hi) / 2
            total = sum(max(0.0, (mid - n['ready_at']) * n['speed']) for n in nodes)
            if total >= N:
                hi = mid
            else:
                lo = mid

        T_star = hi

        # --- Continuous allocation -> rounded to integer ---
        alloc_float = [max(0.0, (T_star - n['ready_at']) * n['speed']) for n in nodes]
        alloc_int   = [math.floor(x) for x in alloc_float]
        remainder   = N - sum(alloc_int)

        # Distribute the remainder to the nodes with the largest fractional part
        fracs = sorted(range(len(nodes)), key=lambda i: -(alloc_float[i] - alloc_int[i]))
        for i in range(remainder):
            alloc_int[fracs[i]] += 1

        # --- Actual makespan after rounding ---
        makespan = max(
            (n['ready_at'] + alloc_int[i] / n['speed'])
            for i, n in enumerate(nodes)
            if alloc_int[i] > 0
        )

        result = {nodes[i]['node_id']: alloc_int[i] for i in range(len(nodes))}
        return makespan, result

    def _estimate_makespan(self, replicas_to_check, nodes_and_replicas, nb_task_per_nodes,
                        job, t_now) -> float:
        #if job.job_id == 4: print([node.node_id for node in replicas_to_check])

        if not replicas_to_check or not nb_task_per_nodes:
            return 0.0

        node_finish_times = []
        t_reference = job.tasks[0].duration

        for i, replica in enumerate(replicas_to_check):
            remaining_time = 0.0
            cn = self.compute_nodes[replica.node_id]
            #for task in job.tasks:
            #    if task.node == replica.node_id and task.status == "Started":
            #        remaining_time = (task.duration/cn.compute_capacity) - (self.env.now - task.starting_time)
            #        break
            
            if replica.transfer_start_time < t_now:
                tr_time = transferCost(self, job.dataset_size, cn.bandwidth)
                tr_end = replica.transfer_start_time + tr_time

                pre_tasks   = nodes_and_replicas.get(replica.node_id, 0)
                alloc_tasks = nb_task_per_nodes[i] if i < len(nb_task_per_nodes) else 0
                total_tasks = pre_tasks + alloc_tasks

                exec_time = total_tasks * (t_reference / cn.compute_capacity)

                finish = (tr_end + exec_time ) - self.env.now 

                if total_tasks <= 0:
                    
                    continue
            else:
                # Remaining time before the transfer finishes on this node
                tr_time     = transferCost(self, job.dataset_size, cn.bandwidth)
                tr_end      = replica.transfer_start_time + tr_time
                wait_time   = max(0.0, tr_end - t_now)  # time to wait from now

                # Total tasks assigned to this node
                pre_tasks   = nodes_and_replicas.get(replica.node_id, 0)
                alloc_tasks = nb_task_per_nodes[i] if i < len(nb_task_per_nodes) else 0
                total_tasks = pre_tasks + alloc_tasks

                # Execution time of the tasks on this node
                exec_time = total_tasks * (t_reference / cn.compute_capacity)

                # Finish time from t_now: wait for the transfer THEN execute
                finish = remaining_time + wait_time + exec_time

            node_finish_times.append(finish)
            if job.job_id == 4:

                print(
                    f"Job {job.job_id} — node {replica.node_id}: tr_rime={tr_time}, exec_time={exec_time}, finish={finish} nb_task={total_tasks} (pre_tasks={pre_tasks}, alloc_tasks={alloc_tasks}) makepsan with this node: {t_now + finish}",
                    #job.job_id, replica.node_id, wait_time, exec_time, finish, pre_tasks, alloc_tasks
                )

                print("Max finish times across nodes:", node_finish_times)
        return max(node_finish_times) if node_finish_times else 0.0

    def select_availables_nodes(self ,k=1):

        free_nodes_id = [node.node_id for i, node in enumerate(self.compute_nodes)]#[i for i in range(len(self.compute_nodes))]
        all_nodes = {node.node_id:node for node in self.compute_nodes}
        for job in self.running_jobs:
            scheduled_task = [task for task in job.tasks if task.status == "Scheduled"]
            for node in list(job.replicas_nodes):
                if node in free_nodes_id and (len(all_nodes[node].queue.items) > 0 or len(all_nodes[node].running_task) > 0
                                              or job.nb_remaining_tasks > 0 or len(scheduled_task)>0 or node in self.actual_transfers.keys()
                                              or not all_nodes[node].free_node or len(all_nodes[node].running_job) != 0 or node in self.actual_transfers.keys()):
                    free_nodes_id.remove(node)


        free_nodes_id = [nid for nid in free_nodes_id if getattr(all_nodes[nid], 'pending_departure', None) is None]

        if k < 0:
            return [all_nodes[node_id] for node_id in free_nodes_id]
        elif len(free_nodes_id) < k:
            return []
        
        k_free_nodes =random.sample(free_nodes_id, k)

        return [all_nodes[node_id] for node_id in k_free_nodes]
    
    def next_free_nodes(self,):
        next_frees_nodes = []
        for job in self.running_jobs:
            job_free_nodes = []
            if job.optimal_replicas_number != None and job.task_execution_time!= None: #and job.optimal_replicas_number <= job.nb_replicas 
                nb_task_per_node = int(len([task for task in job.tasks if task.status == "NotStarted"])//job.nb_replicas) #int((job.nb_tasks - len([task for task in job.tasks if task.status in ["Finished"]]))/job.nb_replicas) #"NotStarted", "Scheduled"
                remaining = len([task for task in job.tasks if task.status == "NotStarted"])%job.nb_replicas #int((job.nb_tasks - len([task for task in job.tasks if task.status in ["Finished"]]))%job.nb_replicas)
                
                if remaining > 0: continue
                
                for node in job.replicas_nodes:
                    delete = False
                    if node not in self.actual_transfers.keys():
                        if len([task for task in job.tasks if task.node == node and task.status == "Started"]) != 0:
                            task_on_node = [task for task in job.tasks if task.node == node and task.status == "Started"][0]
                            task_remaining_time = task_on_node.duration - (self.env.now - task_on_node.starting_time)
                            node_remaining_time = task_remaining_time + (nb_task_per_node*(task_on_node.duration/self.compute_nodes[task_on_node.node].compute_capacity))
                            job_free_nodes.append([node, node_remaining_time, job.job_id])
                        elif len([task for task in job.tasks if task.node == node and task.status == "Scheduled"]) > 0: 
                            delete =True
                    else: delete =True
                    if delete:
                        to_delete = []
                        for i, node_info in enumerate(job_free_nodes):
                            if node_info[0] == node:
                                to_delete.append(i)
                        for i in to_delete:
                            job_free_nodes.pop(i)

                job_free_nodes = sorted(job_free_nodes, key=lambda x: x[1])
                for i in range(len(job_free_nodes)):
                    if i < remaining:
                        job_free_nodes[i][1] = job_free_nodes[i][1] + job.task_execution_time/self.compute_nodes[job_free_nodes[i][0]].compute_capacity

                    add = True
                    for j, node_info in enumerate(next_frees_nodes):
                        if node_info[0] == job_free_nodes[i][0]: 
                            next_frees_nodes.pop(j)
                            add = False
                            break
                            
                    if add: next_frees_nodes.append(copy.deepcopy(job_free_nodes[i]))

        return copy.deepcopy(sorted(next_frees_nodes, key=lambda x: x[1]))
    
    def condition(self, job:Job):
        tasks = [task for task in job.tasks if task.node == job.node_referent and task.status == "Started"] #
        if len(tasks) > 0:
            task_on_refer = tasks[0]
            if task_on_refer.starting_time + task_on_refer.duration/self.compute_nodes[task_on_refer.node].compute_capacity <= self.env.now:
                return 1, True, task_on_refer.duration/ self.compute_nodes[task_on_refer.node].compute_capacity, task_on_refer.duration
            if self.env.now - task_on_refer.starting_time >= job.transfer_time : #and not task_on_refer.validated :
                t_unit = self.env.now - task_on_refer.starting_time
                return 2, True, t_unit, t_unit/self.compute_nodes[task_on_refer.node].compute_capacity

        return None, False, None, None

    def node_to_use_to_starts(self, nb_replica ,job:Job, heterogeneous=False, reference_time=None, f1=1, f2=1, first=False): #f1=0.5, f2=5
        #no node dont do anything
        if not nb_replica: return []
        free_nodes = self.select_availables_nodes(k=-1)
        # A compute-free node that's too small for this dataset doesn't count as
        # "available" for this job (no effect in this artifact: nodes have unlimited
        # capacity, see generate_traces.py - the paper doesn't model a storage
        # constraint).
        free_nodes = [n for n in free_nodes if n.capacity >= job.dataset_size]

        possible_nodes = []
        if self.overlap:
            next_free_nodes = self.next_free_nodes()
            for free_node in next_free_nodes:
                for node in self.compute_nodes:
                    if node.node_id == free_node[0]:
                        free_node[0] = node

            for node in next_free_nodes:
                if node[0].capacity < job.dataset_size:
                    continue
                if (node[1] + 0.2) < transferCost(self, job.dataset_size, node[0].bandwidth): #job.transfer_time:
                    possible_nodes.append(node)
        
        if len(free_nodes) > 0:
            to_add = 0
            if (nb_replica - len(possible_nodes)) < len(free_nodes):
                to_add = nb_replica - len(possible_nodes)
            else:
                to_add = len(free_nodes)
            for node in range(to_add):
                add = True 
                if add: possible_nodes.append((free_nodes[node], 0, -1))
        if first:
            #if len(self.dataset_sizes) == 0:
            #    f1, f2 = 1,0
            #else:
            #    if job.dataset_size >= np.mean(self.dataset_sizes):
            #        f1, f2 = 1,0
            #    else:
            #        f1, f2 = 1,0
            return ComputeNode.sortingByAdaptative(self,possible_nodes)
        return ComputeNode.sortingBy(self,job, possible_nodes, self._config['sorting_criteria'], reference_time=reference_time, f1=f1, f2=f2)

    def nodes_to_use_ovelaping(self, job:Job,first:bool=False):
        
        node_to_sort = []

        for compute_node in self.compute_nodes:
            t_time = transferCost(self, job.dataset_size, compute_node.bandwidth)
            for running_job in self.running_jobs :
                if job.job_id == running_job.job_id: continue

                for replicas in job.replicas:
                    if replicas.node_id == compute_node.node_id:
                        if replicas.makespan - self.env.now + 2 < t_time: 
                            node_to_sort.append((compute_node, max(0,replicas.makespan - self.env.now)))
                            break
        
        free_nodes = self.select_availables_nodes(k=-1)
        
        for node in range(len(free_nodes)):
            add= True
            for node_info in node_to_sort:
                if free_nodes[node].node_id == node_info[0].node_id:
                    add=False
                    break
            if add: node_to_sort.append((free_nodes[node], 0))
        
        if job.job_id==3 and self.env.now < 300:print(f"For job {job.job_id} selection at time {self.env.now}, possible nodes are: ", [(node[0].node_id, node[1]) for node in node_to_sort])

        if first:
            return ComputeNode.sortingByAdaptative(self,node_to_sort)
        return ComputeNode.sortingBy(self,job, node_to_sort, self._config['sorting_criteria'], reference_time=job.tasks[0].duration, f1=1, f2=1)  

    def transfer_data(self, job_id, dataset_size, compute_node, dataset_ready_event,task_id= -1, send_task = False, source_node=None):

        if job_id in self.replicas_locations.keys() and compute_node.node_id in self.replicas_locations[job_id]:
            dataset_ready_event.succeed()
            return

        elif job_id not in self.replicas_locations.keys():
            self.replicas_locations[job_id] = []

        self.replicas_locations[job_id].append(compute_node.node_id)

        topology_factor = heuristic_scheduler.topologyCostFactor(self, source_node, compute_node) if source_node is not None else 1.0
        transfer_time = (dataset_size / compute_node.bandwidth) * topology_factor
        self.actual_transfers[self.compute_nodes[compute_node.node_id].node_id] = (job_id, self.env.now + transfer_time)

        self.compute_nodes[compute_node.node_id].running_job.append(job_id)
        self.compute_nodes[compute_node.node_id].free_node = False

        with compute_node.bandwidth_lock.request() as node_req:
            self.compute_nodes[compute_node.node_id].job_order.append(job_id)
            logger.debug("[%s] Compute-%s: Got new dataset transfert of job %s: duration: %s, dataset_size: %s",
                         self.env.now, compute_node.node_id, job_id, transfer_time, dataset_size)
            yield node_req

            transfer_time = (dataset_size / compute_node.bandwidth) * topology_factor

            self.compute_nodes[compute_node.node_id].running_task.append(-1)
            
            yield self.env.timeout(transfer_time)
            
            end_time = self.env.now
            
            self.compute_nodes[compute_node.node_id].running_task.pop(0)
            self.compute_nodes[compute_node.node_id].datasets.append(job_id)
            dataset_ready_event.succeed()

            self.tracker.log_transfer(job_id, compute_node.node_id, end_time - transfer_time, end_time, dataset_size, task_id=task_id)


            self.compute_nodes[compute_node.node_id].free_node = True  
            if compute_node.node_id in self.actual_transfers.keys(): del self.actual_transfers[self.compute_nodes[compute_node.node_id].node_id]

    def monitorTransfers(self,heterogeneous):
        """Monitor the transfers and remove them when they are finished."""        
        while True:
            yield self.env.timeout(0.4)
            copy_actual_transfers = self.actual_transfers.copy()
            for node in copy_actual_transfers.keys():
                job_info = None
                
                for job in self.running_jobs:
                    if job.job_id == copy_actual_transfers[node][0] and job.node_referent == node:
                        job_info = job
                        break
                if job_info == None: continue
                free_node = self.select_free_nodes(k=-1)#self.nodes_to_use_ovelaping(job_info)#
                if len(free_node)> 0:
                    node_sec = free_node[0]
                    if node_sec[0].node_id != node and copy_actual_transfers[node][1] > self.env.now + (transferCost(self, job_info.dataset_size, node_sec[0].bandwidth) + 0.5):

                        print(f"Alerte Node {node} is not the optimal to start job {job_info.job_id}, switch to {node_sec[0].node_id}")
                        
                        if node_sec[0].node_id not in job_info.replicas_nodes and node in job_info.replicas_nodes:
                            job_info.replicas_nodes.append(node_sec[0].node_id)
                            job_info.replicas_nodes.remove(self.compute_nodes[node].node_id)
                            job_info.idle_replicas.append(self.compute_nodes[node].node_id)
                            transfer_time = transferCost(self,job_info.dataset_size,self.compute_nodes[node_sec[0].node_id].bandwidth) #job.dataset_size / 
                            job_info.transfer_time = transfer_time
                            replica_inst = Replica(job_info.job_id, node_id=self.compute_nodes[node_sec[0].node_id].node_id, data_size=job_info.dataset_size,
                                                    transfer_time=transfer_time,makespan=float("inf"),transfer_start_time=self.env.now)

                            dataset_ready_event = self.env.event()
                            self.env.process(self.transfer_data(job_info.job_id, job_info.dataset_size, self.compute_nodes[node_sec[0].node_id],
                                                    dataset_ready_event))

                            job_info.node_referent = self.compute_nodes[node_sec[0].node_id].node_id
                            job_info.replicas.append(replica_inst)
                            for replicas in job_info.replicas:
                                if replicas.node_id == self.compute_nodes[node].node_id:
                                    job_info.replicas.remove(replicas)
                                    break

                            self.replicas_stats[(job_info.job_id, self.compute_nodes[node_sec[0].node_id].node_id)] = replica_inst
                            if self.compute_nodes[node].new_task and self.compute_nodes[node].new_task.job_id == job_info.job_id:
                                #printprint("Running tasks: ",self.compute_nodes[node].new_task.task_id)
                                task = self.compute_nodes[node].new_task
                                task.dataset_ready_event = dataset_ready_event  # Inject the ready event
                                task.node = self.compute_nodes[node_sec[0].node_id].node_id
                                self.replicas_stats[(job_info.job_id, self.compute_nodes[node_sec[0].node_id].node_id)].nb_tasks += 1
                                self.replicas_stats[(job_info.job_id, self.compute_nodes[node_sec[0].node_id].node_id)].task_execution_time = task.duration
                                task.status = "Scheduled"
                                #self.compute_nodes[node].queue.items.remove(task)
                                self.compute_nodes[node].new_task = None
                                yield self.compute_nodes[node_sec[0].node_id].queue.put(task)
                                #print("[%s] Master: Migrate task %s from job %s to node %s", self.env.now, task.task_id, job_info.job_id, task.node)

                                self.replicas_stats[(job_info.job_id, self.compute_nodes[node].node_id)].nb_tasks -= 1
                                
                            s = 0                
                            for task in self.compute_nodes[node].queue.items:
                                if task.job_id == job_info.job_id:
                                    task.dataset_ready_event = dataset_ready_event  # Inject the ready event
                                    task.node = self.compute_nodes[node_sec[0].node_id].node_id
                                    self.replicas_stats[(job_info.job_id, self.compute_nodes[node_sec[0].node_id].node_id)].nb_tasks += 1
                                    self.replicas_stats[(job_info.job_id, self.compute_nodes[node_sec[0].node_id].node_id)].task_execution_time = task.duration
                                    task.status = "Scheduled"
                                    self.compute_nodes[node].queue.items.remove(task)
                                    yield self.compute_nodes[node_sec[0].node_id].queue.put(task)
                                    #print(f"[{self.env.now}] Master: Migrate task {task.task_id} from job {job_info.job_id} to node {task.node}")

                                    self.replicas_stats[(job_info.job_id, self.compute_nodes[node].node_id)].nb_tasks -= 1
                                    s += 1

                            #if s == 0: print('no task found for job {} on node {}'.format(job_info.job_id, self.compute_nodes[node].node_id))

            if self.finished_jobs == self._config['total_nb_jobs'] and len(self.waiting_jobs) == 0 and  len(self.all_jobs) == self._config['total_nb_jobs'] and len(self.tracker.ongoing_tasks) == 0 and len(self.queue.items) == 0:
                    finished = True
                    for compute_node in self.compute_nodes:  # Be sure that nothing is waiting in any compute queue.
                        if len(compute_node.queue.items) > 0:
                            finished = False
                    if finished:
                        break

    def alloc_branch_and_bound(self, N, free_nodes):
        
        factors = [self.compute_nodes[node.node_id].compute_capacity for node in free_nodes]
        
        m = len(factors)
        best = {'makespan': float('inf'), 'alloc': None}

        # simple lower bound: continuous solution (relaxation)
        inv_sum = sum(f for f in factors)
        T_cont = N / inv_sum
        # lower bound per integer task: max_i ceil( T_cont / f_i )? (useful for light pruning)

        def recurse(i, remaining, current_alloc, current_max_time):
            # i : index of the current node
            # remaining : remaining tasks to assign
            # current_alloc : allocations for nodes 0..i-1
            # current_max_time : max time so far among assigned nodes
            # Pruning: if current_max_time >= best['makespan'], cut this branch
            if current_max_time >= best['makespan']:
                return

            if i == m - 1:
                # send everything to the last node
                alloc = current_alloc + [remaining]
                makespan = max(current_max_time, remaining/factors[i])
                if makespan < best['makespan']:
                    best['makespan'] = makespan
                    best['alloc'] = alloc
                return

            # bounds for x_i : 0..remaining
            # the traversal could be improved by choosing tighter bounds
            for x in range(0, remaining+1):
                new_max = max(current_max_time, x/factors[i])
                recurse(i+1, remaining - x, current_alloc + [x], new_max)

        recurse(0, N, [], 0.0)
        return best['makespan'], best['alloc']  
    
    def optimal_allocation_dict(self, N, free_nodes):
        """
        free_nodes: list of objects with:
            - node.node_id
            - node.compute_capacity  (factor f_i between 0 and 1)
        """

        factors = [self.compute_nodes[node.node_id].compute_capacity for node in free_nodes]
        node_ids = [node.node_id for node in free_nodes]

        # sum of inverses
        inv_sum = sum(1.0 / f for f in factors)

        # optimal continuous makespan
        T_star = N / inv_sum

        # initial integer allocation
        alloc = [int(T_star / f) for f in factors]

        assigned = sum(alloc)
        remaining = N - assigned

        # sort by fastest node (smallest factor)
        order = sorted(range(len(factors)), key=lambda i: factors[i])

        # distribute the remaining tasks
        for i in range(remaining):
            alloc[order[i]] += 1

        # build the final dict
        allocation_dict = {
            node_ids[i]: alloc[i]
            for i in range(len(node_ids))
        }

        # actual makespan (optional)
        makespan = max(factors[i] * alloc[i] for i in range(len(factors)))

        return makespan, allocation_dict

    def selectOptimalNodes(self, job: Job, free_nodes: list, 
                        utility_threshold: float = 1.0) -> UtilityCheckResult:
 
        t_reference = job.tasks[0].duration
        N = job.nb_remaining_tasks
        if N == 0 or not free_nodes:
            return UtilityCheckResult(0, {}, [])

        t_now = self.env.now

        # ── Pre-compute each node's characteristics ───────────────────────
        def node_features(node, is_existing=False):
            tr_time   = transferCost(self, job.dataset_size, node.bandwidth if hasattr(node, 'bandwidth') else self.compute_nodes[node.node_id].bandwidth)
            tr_end    = (node.transfer_start_time if is_existing else t_now) + tr_time
            wait      = max(0.0, tr_end - t_now)
            cap       = self.compute_nodes[node.node_id].compute_capacity
            return {
                'node_id':   node.node_id,
                'wait':      wait,
                'task_time': t_reference / cap,
                'tr_time':   tr_time,
                'existing':  is_existing,
            }

        existing   = [node_features(rep, is_existing=True)  for rep in job.replicas]
        candidates = [node_features(ni[0], is_existing=False) for ni in free_nodes]

        if len(candidates) > 12:
            logger.warning("Too many candidate nodes (%d), truncating to 12.", len(candidates))
            candidates = candidates[:12]
        n_cand = len(candidates)

        # ── Reference makespan (existing replicas only) ───────────────────────────
        if existing:
            makespan_ref, _ = self._dp_allocate(N, existing)
        else:
            # No existing replica -> makespan_ref = +inf (everything still to do)
            makespan_ref = float('inf')

        # ── Compute the utility score for each candidate individually ───────
        utility_scores = {}
        for cand in candidates:
            # Makespan with only this candidate added to the existing ones
            makespan_with, _ = self._dp_allocate(N, existing + [cand])

            gain          = makespan_ref - makespan_with
            transfer_cost = cand['tr_time']

            if transfer_cost <= 0:
                score = float('inf')
            else:
                score = gain / transfer_cost

            utility_scores[cand['node_id']] = score

            logger.debug(
                "Job %d — node %d : gain=%.2f, tr=%.2f, score=%.3f %s",
                job.job_id, cand['node_id'], gain, transfer_cost, score,
                "✓" if score > utility_threshold else "✗"
            )

        # ── Filter out candidates below the utility threshold ─────────────────────────
        useful_candidates = [
            c for c in candidates
            if utility_scores[c['node_id']] > utility_threshold
        ]

        if not useful_candidates:
            logger.debug("Job %d — no candidate node passes the utility threshold %.2f",
                        job.job_id, utility_threshold)
            return UtilityCheckResult(0, {}, [])

        # ── DP over subsets of useful nodes ─────────────────────────────
        n_useful   = len(useful_candidates)
        best_makespan = float('inf')
        best_mask     = 0
        best_alloc    = {}

        for mask in range(1, 1 << n_useful):   # skip mask=0 (no candidate)
            selected = list(existing)
            for b in range(n_useful):
                if mask & (1 << b):
                    selected.append(useful_candidates[b])

            makespan, alloc = self._dp_allocate(N, selected)

            if makespan < best_makespan:
                best_makespan = makespan
                best_mask     = mask
                best_alloc    = {
                    selected[i]['node_id']: alloc[i]
                    for i in range(len(selected))
                }

        # ── Extract the selected nodes ────────────────────────────────────────────
        selected_node_ids = [
            useful_candidates[b]['node_id']
            for b in range(n_useful)
            if best_mask & (1 << b)
        ]

        logger.debug(
            "Job %d — optimal makespan: %.2f (ref: %.2f) with nodes %s",
            job.job_id, best_makespan, makespan_ref, selected_node_ids
        )

        return UtilityCheckResult(
            nb_acceptable_nodes = len(selected_node_ids),
            nodes_and_replicas  = best_alloc,
            acceptable_nodes    = selected_node_ids
        )
    
    def _dp_allocate(self, N: int, nodes: list) -> tuple[float, list]:

        if N == 0:
            return 0.0, [0] * len(nodes)

        m   = len(nodes)
        INF = float('inf')

        # dp[i][n] = min makespan to place n tasks on nodes[0..i]
        # choice[i][n] = number of tasks allocated to nodes[i] in the optimal solution
        dp     = [[INF] * (N + 1) for _ in range(m)]
        choice = [[0]   * (N + 1) for _ in range(m)]

        # Initialization: node 0 receives every task
        for n in range(N + 1):
            ft = nodes[0]['wait'] + n * nodes[0]['task_time'] if n > 0 else 0.0
            dp[0][n]     = ft
            choice[0][n] = n

        # Filling the table
        for i in range(1, m):
            for n in range(N + 1):
                best_mk = INF
                best_k  = 0
                for k in range(n + 1):          # k tasks on nodes[i]
                    ft_i   = nodes[i]['wait'] + k * nodes[i]['task_time'] if k > 0 else 0.0
                    mk_prev = dp[i - 1][n - k]
                    mk      = max(ft_i, mk_prev)
                    if mk < best_mk:
                        best_mk = mk
                        best_k  = k
                dp[i][n]     = best_mk
                choice[i][n] = best_k

        # Reconstructing the allocation
        alloc   = [0] * m
        remain  = N
        for i in range(m - 1, -1, -1):
            alloc[i] = choice[i][remain]
            remain  -= alloc[i]

        return dp[m - 1][N], alloc

    def _is_node_usefulV1(self, replicas_to_check, nodes_and_replicas, nb_task_per_nodes, job) -> bool:
 
        t_reference = job.tasks[0].duration
        # Makespan WITH the candidate node (last in the list sorted by BW)
        makespan_with = max(
            self.compute_nodes[replicas_to_check[i].node_id].compute_capacity
            / nb_task_per_nodes[i]
            for i in range(len(nb_task_per_nodes))
            if nb_task_per_nodes[i] > 0
        ) if any(n > 0 for n in nb_task_per_nodes) else float('inf')

        # Makespan WITHOUT the candidate node — redistribute the tasks over the others
        existing_replicas = replicas_to_check[:-1]
        if not existing_replicas:
            return True  # first node, always useful

        remaining = job.nb_remaining_tasks - sum(nodes_and_replicas.values())
        if remaining <= 0:
            return False  # the existing nodes already cover everything -> not useful

        _, alloc_without = self.alloc_branch_and_bound(
            job.nb_remaining_tasks, existing_replicas
        )

        makespan_without = max(
            self.compute_nodes[existing_replicas[i].node_id].compute_capacity
            / alloc_without[i]
            for i in range(len(alloc_without))
            if alloc_without[i] > 0
        ) if any(n > 0 for n in alloc_without) else float('inf')

        # The new node's transfer must be offset by the makespan gain
        candidate   = replicas_to_check[-1]
        transfer_tr = transferCost(self, job.dataset_size,
                                self.compute_nodes[candidate.node_id].bandwidth)

        gain = makespan_without - makespan_with
        return gain > transfer_tr   # <- the time gain must exceed the transfer cost

