"""
Local, minimal copy of two functions from simulator.py (full repository):
generateHeterogeneousInfrastructureEquilibre and jobs_injector. Extracted here so this
artifact stays self-contained (no import of simulator.py, which pulls in Master variants
not used by the Heuristic approach: master_node.py, master_node_with_overlap.py,
master_node_with_search_algo.py, master_node_with_heterogeneous_nodes_csp.py, plots.py...).
Code unchanged, copied verbatim.
"""
import csv
import json

from job import Job


def generateHeterogeneousInfrastructureEquilibre(config, node_homogeneous=True, path=None):
    """Generate a heterogeneous infrastructure with random bandwidth for each compute node."""
    def extract_bandwidth_cpu(csv_path):
        bandwidth_list = []
        cpu_list = []
        energy_consumption_list = []
        storage_capacity_list = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bandwidth_list.append(float(row["bandwidth"]))
                cpu_list.append(float(row["computation_nodes"]))
                energy_consumption_list.append(float(row["energy_consumption"]))
                storage_capacity_list.append(float(row["storage_capacity"]))

        return bandwidth_list, cpu_list, energy_consumption_list, storage_capacity_list

    if path:
        bandwidth_list, cpu_list, energy_consumption_list, storage_capacity_list = extract_bandwidth_cpu(path)

        nodes_config = [{'bandwidth': bandwidth_list[i], 'computation_nodes': cpu_list[i],
                         'energy_consumption': energy_consumption_list[i],
                         'storage_capacity': storage_capacity_list[i]} for i in range(len(bandwidth_list))]
        return nodes_config

    raise ValueError("path is required (this local copy does not generate infra without a CSV)")


def jobs_injector(env, master, jobs=[], job_file_path=None, config=None):
    """
    if job_file_path is not None, we read the jobs from the file
    else we generate the jobs
    """
    if job_file_path is not None:
        with open(job_file_path, "r", encoding="utf-8") as file:
            jobs = json.load(file)
        config['total_nb_jobs'] = len(jobs)
        current_timeout = 0
        for job in jobs:
            next_arriving_time = job['arriving_time']
            waiting_time = next_arriving_time - current_timeout
            yield env.timeout(waiting_time)
            current_timeout = next_arriving_time
            new_job = Job(job['job_id'], job['task_duration'], job['nb_tasks'], job['dataset_size'],
                          type_of_job=job.get('type_of_job'))
            if getattr(master, "job_classifier", None) is not None:
                new_job.learned_pool = master.job_classifier(new_job)
            yield master.queue.put(new_job)
    else:
        raise ValueError("job_file_path is required (this local copy does not generate jobs on the fly)")
