"""
Trace generator (infrastructure + workload) for evaluating the paper's 3 approaches
"Cost-Aware Replicas Placement and Tasks Scheduling In Geo-Distributed Infrastructure"
(Si-Mohammed et al.) on the same instances: Heuristic (../run_heuristic.py), Offline COP
(../offline-cop/) and Online COP (../online-cop/).

Copy of ../generate_traces.py, placed here so this `traces/` folder is self-sufficient
(you can regenerate its own content without going up to the parent folder). Same logic -
only the output directory differs (see main() below: writes directly here, in the folder
this script lives in, instead of into a `traces/` subfolder). If you change one copy,
port the change to the OTHER one too (or keep only one and make the other a thin wrapper
calling it) to avoid the two drifting apart.

Reproduces the paper's experimental protocol (Section 5.1):
  - Infrastructure with 4 node categories, combining high/low bandwidth and compute power
    (~24/26/24/26% of nodes per category):
        category 0: low VCPU  (1-2.1x),  low bandwidth  (12-128 MB/s   ~ 100Mbit-1Gbit/s)
        category 1: high VCPU (6-10x),   low bandwidth  (12-128 MB/s)
        category 2: low VCPU  (1-2.1x),  high bandwidth (600-800 MB/s ~ 4.8-6.4Gbit/s)
        category 3: high VCPU (6-10x),   high bandwidth (600-800 MB/s)
  - Jobs: number of tasks, task duration and dataset size drawn from realistic discrete
    values; arrivals follow an exponential distribution (lambda_rate = mean inter-arrival
    time, same convention as the rest of the simulator - see generateHeterogeneousInfra-
    structureEquilibre/jobs_injector in simulator.py).
  - The paper's 4 instances: (10 jobs, 50 nodes), (20 jobs, 50 nodes), (20 jobs, 100 nodes),
    (50 jobs, 100 nodes).

Each instance is written to a subfolder inst-{J}J-{N}N/ (next to this script) with two
files:
  - infrastructure.csv: columns bandwidth,computation_nodes,energy_consumption,storage_capacity
  - jobs.json: list of jobs {nb_tasks, task_duration, dataset_size, id_dataset,
    arriving_time, type_of_job, job_id}

Run (from this folder): python3 generate_traces.py
"""
import json
import os
import random

import pandas as pd

# ---------------------------------------------------------------------------
# Protocol parameters (Section 5.1 of the paper)
# ---------------------------------------------------------------------------
SEED = 42

NODE_CATEGORIES = {
    0: {"share": 0.24, "vcpu": (1, 2.1), "bandwidth": (12, 128)},
    1: {"share": 0.26, "vcpu": (6, 10), "bandwidth": (12, 128)},
    2: {"share": 0.24, "vcpu": (1, 2.1), "bandwidth": (600, 800)},
    3: {"share": 0.26, "vcpu": (6, 10), "bandwidth": (600, 800)},
}

MIN_TASKS_PER_JOB = 1
MAX_TASKS_PER_JOB = 20
TASK_DURATIONS_SEC = [10, 50, 70, 80, 100, 120, 140, 150, 180]
DATASET_SIZES_MB = [1024, 2048, 4096, 5120, 7168, 10240, 20480, 40960]
ENERGY_CONSUMPTION_RANGE = (0.1, 2.1)
# The paper does NOT address a storage constraint: nodes are assumed to have UNLIMITED
# capacity (no notion of "node too small for a dataset" in the paper's model - Section
# 4.2, only utility + acceleration factor). storage_capacity is therefore fixed to a
# value deliberately far above any possible dataset, so this filter (present in the
# engine for other uses of the repo) is never a limiting factor here.
UNLIMITED_STORAGE_CAPACITY_MB = 10 ** 9

# (nb_jobs, nb_nodes, lambda_rate) - the 4 instances evaluated in the paper (Section 4.2)
PAPER_INSTANCES = [
    (10, 50, 100),
    (20, 50, 100),
    (20, 100, 100),
    (50, 100, 100),
]


def generate_infrastructure(nb_nodes: int, seed: int = SEED) -> list[dict]:
    """Generates nb_nodes nodes spread across the 4 categories above. Returns a list of
    dicts {bandwidth, computation_nodes, energy_consumption, storage_capacity} - same
    schema as simulator.generateHeterogeneousInfrastructureEquilibre(path=...)."""
    random.seed(seed)
    nodes = []
    assigned = 0
    categories = sorted(NODE_CATEGORIES)
    for idx, cat in enumerate(categories):
        spec = NODE_CATEGORIES[cat]
        if idx == len(categories) - 1:
            count = nb_nodes - assigned  # the remainder, to always total nb_nodes
        else:
            count = round(spec["share"] * nb_nodes)
            assigned += count
        for _ in range(count):
            # Draw order MATTERS (same random.* calls in the same order as the original
            # notebook, to stay bit-for-bit reproducible at a fixed seed): vcpu, then
            # bandwidth, then energy. No random draw for storage - fixed, unlimited
            # capacity (the paper doesn't model it, see above).
            vcpu = random.uniform(*spec["vcpu"])
            bandwidth = random.randint(*spec["bandwidth"])
            energy = random.uniform(*ENERGY_CONSUMPTION_RANGE)
            nodes.append({
                "bandwidth": bandwidth,
                "computation_nodes": vcpu,
                "energy_consumption": energy,
                "storage_capacity": UNLIMITED_STORAGE_CAPACITY_MB,
            })
    return nodes


def save_infrastructure_csv(nodes: list[dict], path: str) -> None:
    df = pd.DataFrame(nodes, columns=["bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"])
    df.to_csv(path, index=False)


def generate_arrival_times(nb_jobs: int, lambda_rate: float, seed: int = SEED) -> list[float]:
    """lambda_rate = MEAN inter-arrival time (seconds), same convention as the rest of
    the simulator (jobsInjectorBasedOnLambdaPoisson in simulator.py): the smaller
    lambda_rate is, the higher the load."""
    random.seed(seed)
    t = 0.0
    times = []
    for _ in range(nb_jobs):
        t += random.expovariate(1 / lambda_rate)
        times.append(t)
    return times


def generate_jobs(nb_jobs: int, lambda_rate: float, seed: int = SEED) -> list[dict]:
    random.seed(seed)
    arrival_times = generate_arrival_times(nb_jobs, lambda_rate, seed=seed)

    jobs = []
    for job_id in range(nb_jobs):
        jobs.append({
            "nb_tasks": random.randint(MIN_TASKS_PER_JOB, MAX_TASKS_PER_JOB),
            "task_duration": random.choice(TASK_DURATIONS_SEC),
            "dataset_size": random.choice(DATASET_SIZES_MB),
            "id_dataset": job_id,
            "arriving_time": arrival_times[job_id],
            "type_of_job": None,
            "job_id": job_id,
        })
    return jobs


def generate_instance(nb_jobs: int, nb_nodes: int, lambda_rate: float,
                       output_dir: str, seed: int = SEED) -> str:
    """Generates one complete instance (infrastructure.csv + jobs.json) in
    output_dir/inst-{nb_jobs}J-{nb_nodes}N/. Returns the path of the created subfolder."""
    inst_dir = os.path.join(output_dir, f"inst-{nb_jobs}J-{nb_nodes}N")
    os.makedirs(inst_dir, exist_ok=True)

    nodes = generate_infrastructure(nb_nodes, seed=seed)
    save_infrastructure_csv(nodes, os.path.join(inst_dir, "infrastructure.csv"))

    jobs = generate_jobs(nb_jobs, lambda_rate, seed=seed)
    with open(os.path.join(inst_dir, "jobs.json"), "w") as f:
        json.dump(jobs, f, indent=2)

    return inst_dir


def main():
    # Only difference from ../generate_traces.py: writes directly into this script's own
    # folder (already named `traces/`), not into a `traces/traces/` subfolder.
    output_dir = os.path.dirname(os.path.abspath(__file__))

    for nb_jobs, nb_nodes, lambda_rate in PAPER_INSTANCES:
        inst_dir = generate_instance(nb_jobs, nb_nodes, lambda_rate, output_dir)
        print(f"[{nb_jobs}J-{nb_nodes}N] lambda_rate={lambda_rate}s -> {inst_dir}")


if __name__ == "__main__":
    main()
