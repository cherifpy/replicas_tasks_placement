"""
Decision logic for the paper's Heuristic approach (Section 4.2: utility function
+ acceleration factor, threshold `sigma`): builds and scores candidates in the "free
nodes" loop, then applies the selected action.

Renamed from storage_capacity_scheduler.py (the name in the full repo, where this module
also handles a per-node storage capacity constraint and preemption strategies to enforce
it - migration/eviction/waiting for an occupant, compensating replica, NSGA. Removed from
this artifact: the paper does NOT model a storage constraint (nodes have unlimited
capacity, see generate_traces.py) and its Heuristic approach uses neither migration nor
preemption - only the decision to ADD a replica on a genuinely free node matters, hence
the new name).

All functions take the master (`UtilityBasedHeterogeneousApproach`) as their first
parameter, the same way `transferCost` or `ComputeNode.sortingBy` already do elsewhere
in this repo.
"""
import copy
import logging
import os
import random

import numpy as np
import pandas as pd

from job import Replica
from compute_node import ComputeNode
from utils.searchAlgo import sortReplicasByNodesBw

logger = logging.getLogger(__name__)


def transferCost(master, dataset_size, node_bw=None):
    """Duplicated verbatim from the module-level helper present in every master_node_*.py."""
    bw = node_bw if node_bw else master._config['compute_node_bw_MBps']
    return dataset_size / bw


def LibererNoeud(node):
    """The node's active dataset has finished its work here (last task of the job that
    was occupying it just completed): the node becomes available again. Used by a
    node's normal lifecycle (master_node_with_heterogeneous_nodes.sendNoStartedJobsOrTasks/
    reschedulTasks), not just by preemption - kept even in this trimmed-down version
    without migration/eviction."""
    node.current_dataset = None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
# Two possible sources for the cost between two nodes:
#   - a REAL N×N bandwidth matrix (MBps, not an abstract factor), loaded from a CSV
#     (`master.topology_matrix`, see load_topology_matrix/generate_rack_topology_matrix
#     below) - the normal case as soon as `config['topology_matrix_path']` points to an
#     existing file. Generated with the same distribution (random.uniform over
#     min/max_compute_node_bw_MBps) as each node's own bandwidth, not arbitrary values;
#   - failing that (no file configured, older experiments), the old synthetic ring based
#     on the node_id gap (no real notion of topology, just "nodes with nearby ids are
#     assumed to be close") - kept so nothing that existed before the matrix was added
#     breaks.

def _ringHopDistance(node_id_a, node_id_b, ring_size):
    d = abs(node_id_a - node_id_b)
    return min(d, ring_size - d)


def _ringTopologyCostFactor(master, source_node, dest_node):
    ring_size = len(master.compute_nodes)
    if ring_size <= 2:
        return 1.0
    max_hop = ring_size // 2
    hop = _ringHopDistance(source_node.node_id, dest_node.node_id, ring_size)
    penalty = max(1.0, master._config.get('inter_node_penalty', 10.0))
    min_factor = 1.0 / penalty
    return min_factor + (1.0 - min_factor) * (hop - 1) / max(1, max_hop - 1)


def topologyCostFactor(master, source_node, dest_node):
    """Multiplicative factor applied to the base transfer cost (dataset_size /
    dest_node.bandwidth) between `source_node` (an already-existing copy of the dataset)
    and `dest_node`. 1.0 (no discount) if source_node is None - identical to the
    historical behavior (transferCost only depends on the destination).

    If a real bandwidth matrix has been loaded (`master.topology_matrix`, values in
    MBps, see load_topology_matrix), the factor is derived so that the final result
    (base * factor) gives exactly dataset_size / link_bandwidth - NOT dataset_size /
    dest.bandwidth: `factor = dest.bandwidth / link_bandwidth`. Unlike the old ring,
    this factor is no longer bounded to [1/inter_node_penalty, 1.0]: a genuinely slow
    inter-rack link can cost more than ignoring topology altogether, which is the point
    (a real network bottleneck, not just a proximity bonus).

    With no matrix loaded, falls back to the old synthetic ring based on node_id
    (_ringTopologyCostFactor, bounded as before)."""
    if source_node is None or source_node.node_id == dest_node.node_id:
        return 1.0
    matrix = getattr(master, 'topology_matrix', None)
    if matrix is not None:
        link_bandwidth = float(matrix[source_node.node_id][dest_node.node_id])
        return dest_node.bandwidth / link_bandwidth
    return _ringTopologyCostFactor(master, source_node, dest_node)


def load_topology_matrix(path):
    """Loads an N×N bandwidth matrix (MBps) from a CSV (row index=node_id,
    columns=node_id). Returns None if `path` is empty/missing (no topology configured -
    unchanged default behavior, falls back to the old ring)."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(int)
    df = df.sort_index()[sorted(df.columns)]
    return df.values


def generate_rack_topology_matrix(nb_nodes, nb_racks=5, min_bw=25.0, max_bw=1280.0,
                                   same_rack_bw_floor_ratio=0.6, seed=None):
    """Generates a realistic default N×N bandwidth matrix (MBps), in per-rack blocks:
    nodes are split into `nb_racks` equally-sized racks (random assignment, not
    contiguous on node_id - so as not to silently correlate rack with node
    characteristics generated elsewhere in the same order).

    Each pair draws its bandwidth with `random.uniform`, EXACTLY like each node's own
    bandwidth (min_compute_node_bw_MBps/max_compute_node_bw_MBps), not an arbitrary
    factor:
      - same rack (fast local link): uniform(min_bw + same_rack_bw_floor_ratio *
        (max_bw - min_bw), max_bw) - the top of the range;
      - different racks (inter-rack link): uniform(min_bw, max_bw) - the full range,
        like a normal node link, potentially a bottleneck.

    Returns (matrix, rack_of_node) - rack_of_node exposed for logging/inspection, not
    used elsewhere. Symmetric matrix (link treated as bidirectional, same simplification
    as the rest of the repo)."""
    rng = random.Random(seed)
    node_ids = list(range(nb_nodes))
    rng.shuffle(node_ids)
    rack_of_node = {}
    for i, node_id in enumerate(node_ids):
        rack_of_node[node_id] = i % nb_racks

    same_rack_floor = min_bw + same_rack_bw_floor_ratio * (max_bw - min_bw)

    matrix = np.zeros((nb_nodes, nb_nodes), dtype=float)
    for a in range(nb_nodes):
        for b in range(a, nb_nodes):
            if a == b:
                bw = max_bw
            elif rack_of_node[a] == rack_of_node[b]:
                bw = rng.uniform(same_rack_floor, max_bw)
            else:
                bw = rng.uniform(min_bw, max_bw)
            matrix[a][b] = bw
            matrix[b][a] = bw
    return matrix, rack_of_node


def save_topology_matrix(matrix, path):
    """Writes a topology matrix (numpy array N×N, bandwidth in MBps) in the CSV format
    expected by load_topology_matrix (index=columns=node_id 0..N-1)."""
    n = matrix.shape[0]
    df = pd.DataFrame(matrix, index=range(n), columns=range(n))
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df.to_csv(path)


def transferCostFromSource(master, dataset_size, source_node, dest_node):
    """Like transferCost, but accounts for the topological proximity of an
    already-existing copy of the dataset (source_node) - see topologyCostFactor.
    source_node=None reproduces transferCost exactly (no existing nearby copy, as is
    still the case today for the first placement and normal replica addition, never
    touched)."""
    base = transferCost(master, dataset_size, dest_node.bandwidth)
    return base * topologyCostFactor(master, source_node, dest_node)


def _bestExistingSource(master, job, dest_node, exclude_node_id=None):
    """Looks, among `job`'s already-existing replicas (excluding exclude_node_id and
    dest_node itself), for the cheapest one to reach for dest_node (real bandwidth/
    topology if `master.topology_matrix` is loaded, otherwise the old ring - see
    transferCostFromSource/topologyCostFactor). None if `job` has no other replica yet -
    transfer "from nothing", like a first placement, never changed.

    Multi-source: a job with several existing replicas should be able to pull a new copy
    from ANY of them, not just from the one currently being evicted - we take the
    cheapest of all of them, not an arbitrarily fixed source.

    `multi_source_replicas` (config, default False - a deliberate rollback): a
    kill switch that fully disables this search - always returns None, so every
    transfer restarts "from the master" as before (transferCost/transfer_data with no
    source, exact historical behavior)."""
    if not master._config.get('multi_source_replicas', False):
        return None
    sources = [
        master.compute_nodes[r.node_id] for r in job.replicas
        if r.node_id != exclude_node_id and r.node_id != dest_node.node_id
    ]
    if not sources:
        return None
    return min(sources, key=lambda s: transferCostFromSource(master, job.dataset_size, s, dest_node))


# ---------------------------------------------------------------------------
# Selection: replaces the historical body of checkUtilityProblem
# ---------------------------------------------------------------------------

def SelectionnerMeilleurNoeud(master, job, free_nodes):
    """Determines the optimal number of nodes to allocate to a job, applying the
    paper's utility/acceleration test (eq. 17-18: utility_ratio <= threshold, threshold
    = improvement / sigma) to every node in `free_nodes` that is genuinely free. No
    storage constraint (the paper doesn't model one, see generate_traces.py) and no
    preemption/migration (out of scope for the Heuristic approach) - this is the
    trimmed-down version of this module, see the file-level docstring."""
    from master_node_with_heterogeneous_nodes import UtilityCheckResult

    replicas_to_add = []
    acceptable_nodes = []
    node_actions = {}
    last_valid_distribution = {}
    makespan_of_each_replicas = {}

    current_best_makespan = job.actual_makespan

    for node in (info[0] for info in free_nodes):

        if job.dataset_size > node.capacity:
            # Node permanently excluded as a candidate for this job, even if it's free.
            continue

        if node.current_dataset is not None and node.current_dataset != job.job_id:
            # "Free" in the sense of select_availables_nodes, but already occupied by
            # another job: we simply skip this node, as if it were unavailable - we
            # only replicate onto genuinely free nodes (no preemption/migration).
            continue

        # Genuinely free node: same logic as the historical checkUtilityProblem, plus
        # `job`'s best existing source (multi-source, see _bestExistingSource) instead
        # of always estimating "from nothing".
        transfer_time = transferCostFromSource(master, job.dataset_size, _bestExistingSource(master, job, node), node)
        candidate_replica = Replica(
            job.job_id,
            node_id=node.node_id,
            data_size=job.dataset_size,
            transfer_time=transfer_time,
            makespan=0,
            transfer_start_time=master.env.now
        )

        all_replicas = job.replicas + replicas_to_add + [candidate_replica]
        replicas_to_check = sortReplicasByNodesBw(all_replicas, master)

        nodes_and_replicas, all_executed = master._compute_pre_executable_tasks(replicas_to_check, job)

        if all_executed >= job.nb_remaining_tasks:
            return UtilityCheckResult(
                len(acceptable_nodes), last_valid_distribution, acceptable_nodes,
                current_best_makespan, makespan_of_each_replicas, node_actions
            )

        max_makespan, affected_tasks, candidate_makespan_per_replicas = master.allocation_of_tasks(job, all_replicas)
        candidate_makespan = max_makespan

        improvement = (
            (current_best_makespan - candidate_makespan) / current_best_makespan
            if current_best_makespan - candidate_makespan > 0 else 0
        )

        if master._config['with_acceleration']:
            seuil = 1 * (improvement / master._config['acceleration_threshold'])
        else:
            seuil = 1

        # 0 tasks assigned = worst possible case, not a division by zero.
        if affected_tasks[candidate_replica.node_id] <= 0:
            break

        if transfer_time / (affected_tasks[candidate_replica.node_id] * job.tasks[0].duration / master.compute_nodes[candidate_replica.node_id].compute_capacity) > seuil:
            break

        acceptable_nodes.append(node.node_id)
        node_actions[node.node_id] = {'action': 'insert'}
        replicas_to_add.append(candidate_replica)
        last_valid_distribution = copy.copy(nodes_and_replicas)
        current_best_makespan = candidate_makespan
        makespan_of_each_replicas = candidate_makespan_per_replicas

    return UtilityCheckResult(
        len(acceptable_nodes), last_valid_distribution, acceptable_nodes,
        current_best_makespan, makespan_of_each_replicas, node_actions
    )


# ---------------------------------------------------------------------------
# Application: carries out the selected action for an accepted node (the only action
# possible in this artifact: 'insert' - no migration/eviction, see the file-level
# docstring)
# ---------------------------------------------------------------------------

def AppliquerSelection(master, job, node, action_info, makespan_peer_replicas):
    """Applies the action selected by SelectionnerMeilleurNoeud for `node` (always
    'insert' in this artifact), then starts the new job's normal transfer. Returns the
    dataset_ready_event to attach to the task, the same way the historical inline block
    in reschedulTasks used to."""
    best_source = _bestExistingSource(master, job, node)
    job.replicas_nodes.append(node.node_id)
    job.transfer_time = transferCostFromSource(master, job.dataset_size, best_source, node)

    dataset_ready_event = master.env.event()
    master.env.process(master.transfer_data(job.job_id, job.dataset_size, node, dataset_ready_event,
                                             source_node=best_source))

    replica_inst = Replica(
        job.job_id, node_id=node.node_id, data_size=job.dataset_size,
        transfer_time=job.transfer_time, makespan=makespan_peer_replicas.get(node.node_id, float('inf')),
        transfer_start_time=master.env.now
    )
    job.replicas.append(replica_inst)
    master.replicas_stats[(job.job_id, node.node_id)] = replica_inst
    job.nb_replicas += 1

    node.current_dataset = job.job_id

    return dataset_ready_event
