# Artifact — Cost-Aware Replicas Placement and Tasks Scheduling In Geo-Distributed Infrastructure

This folder is **100% self-contained**: it has everything needed to reproduce the
3 approaches evaluated in the paper — a shared trace generator
(infrastructure + workload, `generate_traces.py`) and, for each approach, the engine
needed to run it on the same instances:

- **Heuristic** (Section 4.2: utility function + acceleration factor, threshold
  `sigma`) — in this folder itself, see `run_heuristic.py` below.
- **Offline COP** (Section 4.1: Choco-solver + LNS, the whole horizon planned in one
  resolution) — subfolder `offline-cop/`, see its own README.
- **Online COP** (Section 4.1: Choco-solver + LNS, replanned on every arrival without
  reconsidering work already started) — subfolder `online-cop/`, see its own
  README.

No file, path, or import reaches outside this folder: it can be sent/copied on its own,
independently of the rest of the repository (Java included — the Offline/Online COP
solvers vendor their own jars, see `offline-cop/lib/` and `online-cop/utils/model/lib/`).

**To run an experiment on any of the 3 approaches with a single command** (instead of
invoking each runner separately): see `exps/run_experiment.py` and `exps/README.md`.

**Model scope (matches the paper)**: the paper does not address a storage constraint —
nodes have **unlimited** capacity (`generate_traces.py` sets `storage_capacity` to a
value deliberately far above any dataset, so this filter of the engine — present for
other uses of the full repository — is never binding here). There is no migration or
preemption: only the decision to ADD a replica counts (utility + acceleration factor,
threshold `sigma`). Unlike the full repository (which also manages a per-node capacity
constraint and preemption strategies to enforce it — migration, deletion, waiting, NSGA),
this artifact does NOT contain that code: `heuristic_scheduler.py` and
`master_node_with_heterogeneous_nodes.py` have been trimmed down to keep only the
Heuristic approach's logic (Section 4.2), and `migration_nsga.py` is not included at all.

## Contents

**Scripts to run:**
- `generate_traces.py` — generates the 4 instances evaluated in the paper (Section 5.1):
  `(10 jobs, 50 nodes)`, `(20 jobs, 50 nodes)`, `(20 jobs, 100 nodes)`,
  `(50 jobs, 100 nodes)`. Cleaned-up, self-contained version of the original notebook
  from the full repository (`simulator/workloads/workloads-100-storage-contrainte/job_genetation.ipynb`).
- `run_heuristic.py` — loads a generated trace and runs the Heuristic on it,
  reports `avg flow time`, `amount of data transferred`, `nb transfers` (the metrics
  from Fig. 2/4/5 and Table 3 of the paper).

**Simulation engine (copied from the full repository):**
`job.py`, `compute_node.py`, `tracker.py`, `utils/searchAlgo.py`, `utils/classifier.py`,
`config.json` unchanged; `heuristic_scheduler.py` and
`master_node_with_heterogeneous_nodes.py` trimmed down (storage/preemption/migration
constraints removed, see above) — see the correspondence table below.

**`engine.py`** — local copy of two functions from `simulator.py` (the full repository's
entry point, which is NOT included here since it also imports Master variants not used
by the Heuristic — overlap, search, CSP, RL...):
`generateHeterogeneousInfrastructureEquilibre` (loads an infrastructure CSV) and
`jobs_injector` (injects jobs from a JSON trace respecting their `arriving_time`).
Code identical to the original, just extracted to avoid these unnecessary dependencies.

`traces/` — created by `generate_traces.py` (can be regenerated at any time, can be
deleted without loss). Also contains a standalone copy of the generator
(`traces/generate_traces.py`, same draws, writes directly into this folder) so instances
can be regenerated without going up to the parent folder — keep both copies in sync if
you modify one (see the header of `traces/generate_traces.py`).

**Requirements**: Python 3.10+, `pip install simpy numpy pandas`.

## Correspondence with the paper

| Paper concept | File / function (in this folder) |
|---|---|
| Utility function (eq. 1) + acceleration factor (eq. 17-18), threshold `sigma` | `heuristic_scheduler.py` → `SelectionnerMeilleurNoeud` (threshold `sigma` = `config['acceleration_threshold']`) |
| Selecting the fastest reference node (eq. 15) | `compute_node.py` → `ComputeNode.sortingByAdaptative` |
| Sorting function `F(n)` over available nodes (eq. 16) | `compute_node.py` → `ComputeNode.sortingBy` (`sorting_criteria="bandwidth_vcpu"`) |
| Master/Workers orchestration, transfers, task allocation | `master_node_with_heterogeneous_nodes.py` → `UtilityBasedHeterogeneousApproach` |
| Generating infrastructure from a CSV | `engine.py` → `generateHeterogeneousInfrastructureEquilibre` |
| Injecting jobs from a JSON trace (respects `arriving_time`) | `engine.py` → `jobs_injector` |

`run_heuristic.py` only assembles these building blocks around a generated trace — zero
reimplementation of the decision logic.

The **Offline COP** and **Online COP** approaches (Section 4.1, Choco-solver/LNS) are in
the `offline-cop/` and `online-cop/` subfolders (see their respective READMEs) — they
use the same traces (`traces/inst-XJ-YN/`, generated once by `generate_traces.py` at the
root of this artifact).

## Usage

```bash
cd simulator/paper-artifact

# 1. Generate the paper's 4 instances into traces/
python3 generate_traces.py

# 2. Run the Heuristic on an instance, at a given sigma
python3 run_heuristic.py traces/inst-20J-50N --sigma 0.05
python3 run_heuristic.py traces/inst-20J-50N --sigma 0.20   # more selective, cf. Fig. 6 of the paper
```

`run_heuristic.py` output (example, `inst-20J-50N`, sigma=0.05, seed=42):

```
            instance: inst-20J-50N
               sigma: 0.05
    nb_jobs_finished: 20
     avg_flow_time_s: 673.62
   amount_of_data_MB: 431104
        nb_transfers: 82
```

As `sigma` increases, the number of transfers and the data volume decrease (stricter
criterion, less replication) — same trend as Fig. 6b/6c of the paper.

## Format of the generated traces

`traces/inst-{J}J-{N}N/infrastructure.csv` — one line per node:

```
bandwidth,computation_nodes,energy_consumption,storage_capacity
```

`bandwidth` (MB/s) and `computation_nodes` (speed factor) follow the paper's 4-category
split (~24/26/24/26% of nodes, high/low bandwidth and compute power combined — see
`NODE_CATEGORIES` in `generate_traces.py`).

`storage_capacity` is always `1_000_000_000` (MB) — a constant deliberately far above
any possible dataset (max 40960 MB), since the paper does not model a storage constraint
(nodes have unlimited capacity). This column only exists because the simulation engine
(shared with the rest of the repository) expects it in the CSV; it is never a limiting
factor here.

`traces/inst-{J}J-{N}N/jobs.json` — a list of jobs:

```json
{"nb_tasks": 6, "task_duration": 140, "dataset_size": 10240,
 "id_dataset": 0, "arriving_time": 102.0, "type_of_job": null, "job_id": 0}
```

`arriving_time` follows an exponential distribution with mean `lambda_rate` (100s by
default for the paper's 4 instances).

## Fidelity note

`generate_traces.py` faithfully reproduces bandwidth/computation_nodes/energy_consumption
and the job format (verified by direct comparison against traces produced by the
original notebook — identical values at the same seed, same order of random draws).

The paper does not address a storage constraint: its model (Section 4.2) only reasons
about utility + acceleration factor, with no notion of "a node too small for a
dataset". `storage_capacity` is therefore set to a constant (`UNLIMITED_STORAGE_CAPACITY_MB
= 10**9`, far above any generated dataset, max 40960 MB) rather than drawn
randomly — this choice is deliberately simpler than a bounded draw would be, and
avoids a pitfall already encountered while developing this artifact: a CONTINUOUS draw
within the dataset-size bounds (`random.randint(1026, 40961)`) makes it near
impossible for any one of a hundred nodes to land exactly on the capacity needed for the
largest dataset (2 values out of ~40000 possible) — verified at runtime, this used to
block jobs indefinitely. With unlimited capacity this risk no longer exists by
construction.

Likewise, the artifact does not model migration or preemption: this code (present in
the full repository — migration/deletion/waiting for an occupant, compensating replica,
NSGA search) has been removed from `heuristic_scheduler.py` and
`master_node_with_heterogeneous_nodes.py`, and `migration_nsga.py` is not included —
only the decision to add a replica (utility + acceleration, threshold `sigma`) is
exercised, matching the scope of the paper's Heuristic approach.

Verified at runtime: the paper's 4 instances complete 100% of their jobs.

| instance | jobs finished | avg flow time (s) | volume (MB) | nb transfers |
|---|---|---|---|---|
| inst-10J-50N | 10/10 | 432.26 | 331 776 | 39 |
| inst-20J-50N | 20/20 | 673.62 | 431 104 | 82 |
| inst-20J-100N | 20/20 | 330.20 | 637 952 | 115 |
| inst-50J-100N | 50/50 | 341.47 | 2 897 920 | 254 |

(sigma=0.05, seed=42, `python3 run_heuristic.py traces/inst-XJ-YN --sigma 0.05`)
