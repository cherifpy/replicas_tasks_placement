"""
Runs the Online COP approach from the paper "Cost-Aware Replicas Placement and Tasks
Scheduling In Geo-Distributed Infrastructure" (Si-Mohammed et al.) on a trace generated
by ../generate_traces.py - the same trace (infrastructure.csv + jobs.json) used by
../run_heuristic.py and ../offline-cop/RunOfflineCOP, to allow a direct comparison of
the 3 approaches on the same instances.

Online COP: on every job arrival, (re)plans in a single shot ALL jobs not yet fully
assigned (the newly arrived one + those already running but with tasks still
NotStarted) via a Choco-solver+LNS model (MainOnline.java, see
utils/model/src/main/MainOnline.java), feeding it only an ESTIMATE of when each node
will free up (`nodesFreeTime` - current transfer/task, no assumption about the future
beyond that). Unlike Offline COP, there is no global recomputation over the whole
horizon: a task that has already started (status "Started") is NEVER reconsidered -
only the assignment of tasks still NotStarted can change from one call to the next.

Copied/adapted from simulator-for-CSP-model/simulator/ (separate Python project, same
paths as the one used to verify the trace format - see README.md):
master_node_online.py (class SchedulingUsingCSPOnline, extracted from
master_node_with_heterogeneous_nodes_csp.py - only the base "Online" variant is kept,
not WarmStart/ThreeStep/Incremental/SemiOnline) and modelCSP_online.py (bridge to
MainOnline.java, extracted from utils/modelCSP.py). Like offline-cop/, no storage
constraint (nodes have unlimited capacity, see ../generate_traces.py).

Usage:
    python3 run_online_cop.py ../traces/inst-10J-50N
    ONLINE_COP_SOLVER_TIME_LIMIT_S=15 python3 run_online_cop.py ../traces/inst-10J-50N

`ONLINE_COP_SOLVER_TIME_LIMIT_S` (default 5): time budget (seconds) granted to EACH call
of the Java solver - a call happens on every (re)planning (at least once per arriving
job, potentially more for a multi-task job while its placement keeps being recomputed).
A larger budget improves the quality of each solution but the total (wall-clock) time
of the simulation grows with the number of calls - keep a modest value (a few seconds)
for a demo, increase it to reproduce results closer to the paper's.
`ONLINE_COP_OBJECTIVE` (default 0): 0 = sum of flow times of the current batch
(comparable to avg_flow_time_s of the other two approaches), 1 = max flow time of the
batch, 2 = flow time of the newly arrived job alone - see modelCSP_online.py.
"""
import argparse
import json
import os
import sys

ARTIFACT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ARTIFACT_ROOT)  # everything is local to this folder - see README.md

import numpy as np
import simpy

from job import Job
from tracker import Tracker
from compute_node import ComputeNode
from master_node_online import SchedulingUsingCSPOnline


def load_infrastructure_csv(path):
    import csv
    nodes = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append({
                "bandwidth": float(row["bandwidth"]),
                "computation_nodes": float(row["computation_nodes"]),
                "energy_consumption": float(row.get("energy_consumption", 1.0)),
                "storage_capacity": float(row["storage_capacity"]) if "storage_capacity" in row else float("inf"),
            })
    return nodes


def jobs_injector(env, master, jobs_file_path):
    """Copy of the file-based jobs_injector from
    simulator-for-CSP-model/simulator/simulator.py - same 4 positional fields as
    classes.job.Job (job_id, task_duration, nb_tasks, dataset_size), `arriving_time`
    handled here as a delay between successive arrivals."""
    with open(jobs_file_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    current_timeout = 0
    for job in jobs:
        next_arriving_time = job['arriving_time']
        waiting_time = next_arriving_time - current_timeout
        yield env.timeout(waiting_time)
        current_timeout = next_arriving_time
        yield master.queue.put(Job(job['job_id'], job['task_duration'], job['nb_tasks'], job['dataset_size']))


def run_instance(instance_dir: str, until: float = 200000, solver_time_limit_s: float = None):
    infra_csv = os.path.join(instance_dir, "infrastructure.csv")
    jobs_file = os.path.join(instance_dir, "jobs.json")
    with open(jobs_file) as f:
        nb_jobs = len(json.load(f))

    if solver_time_limit_s is None:
        solver_time_limit_s = float(os.environ.get("ONLINE_COP_SOLVER_TIME_LIMIT_S", 5))

    nodes_config = load_infrastructure_csv(infra_csv)

    config = {
        "total_nb_jobs": nb_jobs,
        "total_nb_compute_nodes": len(nodes_config),
        "threshold": 1,
        "overlap": False,
        "homogeneous": False,
        "compute_node_bw_MBps": 384,
        "compute_node_latency_ms": 1,
        "solver_time_limit_s": solver_time_limit_s,
    }

    env = simpy.Environment()
    tracker = Tracker(env)
    master = SchedulingUsingCSPOnline(env, [], tracker, config, overlap=False)
    compute_nodes = [
        ComputeNode(env, i, master, bandwidth=nodes_config[i]["bandwidth"],
                    compute_capacity=nodes_config[i]["computation_nodes"],
                    energy_consumption=nodes_config[i]["energy_consumption"],
                    storage_capacity=nodes_config[i].get("storage_capacity", float("inf")))
        for i in range(len(nodes_config))
    ]
    master.compute_nodes = compute_nodes
    master.nb_nodes = len(compute_nodes)

    env.process(master.receiveJobs())
    env.process(master.schedulingNewJob())
    env.process(master.scheduling())
    env.process(master.checkOnJobs())
    for node in compute_nodes:
        env.process(node.processTasks())
    env.process(jobs_injector(env, master, jobs_file))

    env.run(until=until)

    if master.finished_jobs < nb_jobs:
        print(f"WARNING: only {master.finished_jobs}/{nb_jobs} jobs finished "
              f"(increase --until if needed)")

    flow_times = [j["finishing_time"] - j["arriving_time"] for j in tracker.stats_on_jobs]
    return {
        "instance": os.path.basename(instance_dir.rstrip("/")),
        "nb_jobs_finished": len(flow_times),
        "avg_flow_time_s": float(np.mean(flow_times)) if flow_times else float("nan"),
        "amount_of_data_MB": tracker.total_nb_transferred_bytes,
        "nb_transfers": tracker.total_nb_transfers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance_dir", help="Instance directory (e.g. ../traces/inst-20J-50N)")
    parser.add_argument("--until", type=float, default=200000, help="Max simulation horizon (s)")
    parser.add_argument("--solver-time-limit", type=float, default=None,
                         help="Time budget (s) per Java solver call (default: $ONLINE_COP_SOLVER_TIME_LIMIT_S or 5)")
    args = parser.parse_args()

    result = run_instance(args.instance_dir, args.until, args.solver_time_limit)
    for k, v in result.items():
        print(f"{k:>20}: {v}")


if __name__ == "__main__":
    main()
