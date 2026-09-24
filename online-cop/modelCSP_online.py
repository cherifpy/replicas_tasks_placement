"""
Python -> Java bridge for the Online COP approach: on every scheduling tick,
serializes the current state (jobs to (re)place, when each node becomes free) into
exchange files, invokes the Java solver (MainOnline, Choco-solver + LNS - see
utils/model/src/main/MainOnline.java), then reads its plan back (transfers/tasks/drops).

Lightweight version (for this artifact) of
simulator-for-CSP-model/simulator/utils/modelCSP.py (separate repo): only keeps the path
used by SchedulingUsingCSPOnline (schedulingUsingJavaCSP/toDict/loadDeletions/
sortSolution) - the MiniZinc fallback (startMinizincModel), the pure-Python solver
(onLineSchedulingUsingCSP, never actually called even in the original repo), and the
inputs/outputs specific to other Java entry points (MainIncremental, MainOnlineMultiObj...)
that MainOnline.java doesn't read have been removed.
"""
import csv
import os
import subprocess

import pandas as pd

ONLINE_COP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ONLINE_COP_ROOT, "utils", "model")

# 0 = sum of flow times (every job in the batch) - comparable to avg_flow_time_s of the
# other two approaches (Heuristic, Offline COP); 1 = max flow time (historical default of
# the original Java file); 2 = flow time of the newly arrived job alone. Overridable via
# the ONLINE_COP_OBJECTIVE environment variable.
DEFAULT_OBJECTIVE_CHOICE = 0


def sortSolution(transfers, works):
    for key, item in transfers.items():
        transfers[key] = sorted(item, key=lambda x: x[2])
    for key, item in works.items():
        works[key] = sorted(item, key=lambda x: x[3])
    return transfers, works


def schedulingUsingJavaCSP(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list, scheduling_start_time=None):
    """Serializes the current state, invokes MainOnline (javac then java, recompiled on
    every call - same as the original repo), reads its plan back. Returns (transfers,
    works, deletions), or ({}, {}, {}) if no solution was found."""
    import json

    jobs = sorted(jobs, key=lambda x: x.job_id)

    # replicas_locations.json: for each job in the batch (index = job_id), the list of
    # nodes where its dataset is already (or currently being) transferred - a transfer
    # still in progress (not finished yet, so absent from replicas_locations) also counts
    # as already resident: otherwise a later solve would see it as a freely replannable
    # candidate even though that transfer is going to land regardless.
    ongoing_nodes_by_job = {}
    for key, ongoing in master_node.ongoing_transfers.items():
        if ongoing is not None:
            ongoing_job_id, ongoing_node_id = ongoing[0], ongoing[1]
            ongoing_nodes_by_job.setdefault(ongoing_job_id, set()).add(ongoing_node_id)

    matrix = []
    for job in jobs:
        resident_nodes = list(replicas_locations.get(job.job_id, []))
        for node_id in ongoing_nodes_by_job.get(job.job_id, ()):
            if node_id not in resident_nodes:
                resident_nodes.append(node_id)
        matrix.append(resident_nodes)
    with open(os.path.join(MODEL_DIR, "inputs", "replicas_locations.json"), "w") as f:
        json.dump(matrix, f)

    jobs_data = []
    for job in jobs:
        nb_not_started = len([t for t in job.tasks if t.status == "NotStarted"])
        if nb_not_started > 0:
            jobs_data.append({
                "job_id": job.job_id,
                "dataset_size": job.dataset_size,
                "nb_tasks": nb_not_started,
                "task_duration": job.tasks[0].duration,
                "timelasped": int(master_node.env.now - job.arriving_time) + 1,
                # Lower bound (LOCAL frame for this solve, 0 = "now") on this job's
                # earliest possible transfer - 0 for any job that has already arrived (the
                # normal case), non-zero only if a future caller batches jobs with
                # different real arrival times into a single solve.
                "job_arriving_time": max(0, job.arriving_time - master_node.env.now),
            })
    jobs_data = sorted(jobs_data, key=lambda x: x['job_id'])
    pd.DataFrame(jobs_data).to_json(os.path.join(MODEL_DIR, "inputs", "jobs.json"), orient="records", indent=4)

    # Solver time budget per call - configurable via config['solver_time_limit_s']
    # (default 120s on the Java side if this file is missing/unreadable).
    solver_time_limit_s = master_node._config.get('solver_time_limit_s', 120)
    with open(os.path.join(MODEL_DIR, "inputs", "solver_time_limit.txt"), "w") as f:
        f.write(str(int(solver_time_limit_s)))

    objective_choice = int(os.environ.get("ONLINE_COP_OBJECTIVE", DEFAULT_OBJECTIVE_CHOICE))
    with open(os.path.join(MODEL_DIR, "inputs", "objective_choice.txt"), "w") as f:
        f.write(str(objective_choice))

    nodes_list = []
    for node_id, node in enumerate(master_node.compute_nodes):
        storage_capacity = getattr(node, 'storage_capacity', float('inf'))
        nodes_list.append({
            "node_id": node_id,
            "bandwidth": node.bandwidth,
            "compute_capacity": node.compute_capacity,
            "free_time": nodes_free_time[node_id],
            # JSON/Java have no infinity: cap at a value the solver treats as unlimited -
            # never binding here (the paper doesn't model a storage constraint, see
            # ../generate_traces.py).
            "storage_capacity": int(storage_capacity) if storage_capacity != float('inf') else 2 ** 30,
        })
    pd.DataFrame(nodes_list).to_json(os.path.join(MODEL_DIR, "inputs", "nodes.json"), orient="records", indent=4)

    java_main_class = getattr(master_node, 'java_main_class', 'MainOnline')

    compile_result = subprocess.run(
        [
            "javac", "-cp", os.path.join(MODEL_DIR, "lib", "*"),
            "-d", os.path.join(MODEL_DIR, "bin"),
            os.path.join(MODEL_DIR, "src", "main", f"{java_main_class}.java"),
        ],
        capture_output=True, text=True, cwd=ONLINE_COP_ROOT,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(
            f"javac failed to compile {java_main_class}.java (exit code {compile_result.returncode}).\n"
            f"--- stderr ---\n{compile_result.stderr}"
        )

    run_result = subprocess.run(
        [
            "java",
            # Avoids a startup failure ("graal_create_isolate error") on some JDKs (25+)
            # in memory-constrained environments - Choco doesn't need anything from Graal.
            "-XX:+UnlockExperimentalVMOptions", "-XX:-UseJVMCICompiler",
            "-cp", os.path.join(MODEL_DIR, "bin") + ":" + os.path.join(MODEL_DIR, "lib", "*"),
            f"main.{java_main_class}",
        ],
        capture_output=True, text=True, cwd=ONLINE_COP_ROOT,
    )
    if run_result.returncode != 0:
        raise RuntimeError(
            f"Java scheduler ({java_main_class}) exited with code {run_result.returncode}.\n"
            f"--- stdout ---\n{run_result.stdout}\n--- stderr ---\n{run_result.stderr}"
        )

    model_output_path = os.path.join(MODEL_DIR, "outputs")
    job_ids = [job['job_id'] for job in jobs_data]
    works = toDict(f"{model_output_path}/works.csv", nb_nodes=len(master_node.compute_nodes), job_list=job_ids, master_node=master_node)
    transfers = toDict(f"{model_output_path}/transfers.csv", nb_nodes=len(master_node.compute_nodes), job_list=job_ids, master_node=master_node)
    deletions = loadDeletions(f"{model_output_path}/deletions.csv", nb_nodes=len(master_node.compute_nodes), job_list=job_ids, master_node=master_node)

    # toDict()/loadDeletions() always pre-populate one key per node (even for an empty
    # CSV): an entirely empty works dict is therefore the real "solver found nothing" signal.
    if not any(len(v) > 0 for v in works.values()):
        return {}, {}, {}

    transfers, works = sortSolution(transfers, works)
    return transfers, works, deletions


def toDict(path_to_csv, nb_nodes=None, job_list=None, master_node=None):
    dict_info = {f"node_{node_id}": [] for node_id in range(nb_nodes if nb_nodes is not None else 100)}

    with open(path_to_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        now = master_node.env.now
        for row in reader:
            job_index = int(row["job_index"])
            start_time = int(row["start_time"])
            end_time = int(row["end_time"])
            node_index = int(row["node_index"])
            if 'task_index' in row.keys():
                task_index = int(row["task_index"])
                dict_info[f"node_{node_index}"].append(
                    (job_list[job_index], node_index, task_index, now + start_time, now + end_time, end_time - start_time)
                )
            else:
                dict_info[f"node_{node_index}"].append(
                    (job_list[job_index], node_index, now + start_time, now + end_time, end_time - start_time)
                )

    return dict_info


def loadDeletions(path_to_csv, nb_nodes=None, job_list=None, master_node=None):
    """Reads the CSP's abandonment decisions: for each (job, node) it chose to abandon,
    when to actually free that node's storage."""
    dict_info = {f"node_{node_id}": [] for node_id in range(nb_nodes if nb_nodes is not None else 100)}

    with open(path_to_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        now = master_node.env.now
        for row in reader:
            job_index = int(row["job_index"])
            node_index = int(row["node_index"])
            deletion_time = int(row["deletion_time"])
            dict_info[f"node_{node_index}"].append((job_list[job_index], now + deletion_time))

    for key, item in dict_info.items():
        dict_info[key] = sorted(item, key=lambda x: x[1])

    return dict_info
