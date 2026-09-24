import numpy as np
import logging
import math

def quartiles(values):
    if not values:
        return (0, 0, 0)
    q1, q3 = np.percentile(values, [25, 75])
    return (q1, np.median(values), q3, np.mean(values), np.std(values))

def classify(value, q1, q3, mean=None, std=None):
    """Returns 'small', 'medium', 'large', 'very large' based on IQR."""
    if mean:
        if value > mean:
            return "large"
        else:
            return "small"

    iqr = q3 - q1
    if value < q1:
        return "small"
    elif value <= q3:
        return "medium"
    elif value <= q3 + 1.5 * iqr:
        return "large"
    else:
        return "very large"

def choisir_alpha_beta(classification):
    if classification["dataset_size"] in ["large", "very large"]:
        if classification["nb_tasks"] in ["large", "very large"]:
            if classification["task_duration"] in ["large", "very large"]:
                return 1, 0
            elif classification["task_duration"] in ["small", "medium"]:
                return 0.8, 0.2

        elif classification["nb_tasks"] in ["small", "medium"]:
            if classification["task_duration"] in ["large", "very large"]:
                return 0.8, 0.2
            elif classification["task_duration"] in ["small", "medium"]:
                return 1, 0

    elif classification["dataset_size"] in ["small", "medium"]:
        if classification["nb_tasks"] in ["large", "very large"]:
            if classification["task_duration"] in ["large", "very large"]:
                return 0, 1
            elif classification["task_duration"] in ["small", "medium"]:
                return 0.3, 0.7

        elif classification["nb_tasks"] in ["small", "medium"]:
            if classification["task_duration"] in ["large", "very large"]:
                return 0.2, 8
            elif classification["task_duration"] in ["small", "medium"]:
                return 1, 0

    return 1, 0  # balanced default case

def estimatingParams(master_node, job_to_reschedule=None, nb_nodes=0, avg_bandwidth=0, avg_cpu=0):
    job_id = job_to_reschedule.job_id

    # === 1. Collect global stats ===
    nb_jobs = 0
    nb_tasks_all, dataset_sizes, task_durations = [], [], []

    # Job history
    for old_job in master_node.tracker.stats_on_jobs:
        nb_tasks_all.append(old_job["nb_tasks"])
        dataset_sizes.append(old_job["dataset size"])
        task_durations.append(old_job["task_execution_time"])
        nb_jobs += 1

    # Currently running jobs (excluding the target job)
    for running_job in master_node.running_jobs:
        if running_job.job_id != job_id:
            nb_remaining_tasks = len([task for task in running_job.tasks if task.status != "NotStarted"])
            nb_tasks_all.append(nb_remaining_tasks)
            dataset_sizes.append(running_job.dataset_size)
            if running_job.tasks and running_job.tasks[0].duration is not None:
                task_durations.append(running_job.tasks[0].duration)
            nb_jobs += 1

    if nb_jobs == 0:
        return (1, 1)  # not enough data

    q_nb, med_nb, Q3_nb, mean_nb, _ = quartiles(nb_tasks_all)
    q_ds, med_ds, Q3_ds, mean_ds, _ = quartiles(dataset_sizes)
    q_td, med_td, Q3_td, mean_td, _ = quartiles(task_durations)

    job_tasks = len([task for task in job_to_reschedule.tasks if task.status == "NotStarted"])#job_to_reschedule.nb_tasks#if task.status != "NotStarted"
    job_ds    = job_to_reschedule.dataset_size
    job_td    = (job_to_reschedule.tasks[0].duration if job_to_reschedule.tasks and job_to_reschedule.tasks[0].duration is not None else None)

    if job_td is None:
        return (1, 1)  # not enough data

    # build features vector
    list_of_features = [
        job_ds, job_tasks, job_td, nb_nodes/master_node._config['total_nb_compute_nodes'], avg_bandwidth, avg_cpu
    ]

    features = list_of_features[0:master_node.feature_dim]


    # safety fallback
    if not master_node.rl_agent:
        classification = {
            "dataset_size" : classify(job_ds, q_ds, Q3_ds, mean_ds),
            "nb_tasks"     : classify(job_tasks, q_nb, Q3_nb, mean_nb),
            "task_duration": classify(job_td, q_td, Q3_td, mean_td),
        }
        return choisir_alpha_beta(classification)
    """if job_to_reschedule.params:
        return job_to_reschedule.params"""
    alpha, beta = master_node.rl_agent.select_action(features, job_to_reschedule.job_id)
    #job_to_reschedule.params = (alpha, beta)
    #print("we selected action:", (alpha, beta), "for job", job_id)
    return alpha, beta


def adaptingTheta(master_node):
    """
        this function adapte the threshold of all running jobs based of the lambda variability
        if lambda increase the threshold also increase with the same ratio
        it did the same for decreasing
    """
    lambda_before = master_node._config['lambda_rate']
    while True:

        yield master_node.env.timeout(0.01)

        if lambda_before != master_node._config['lambda_rate']:
            print("lambda diff")
            lambda_now = master_node._config['lambda_rate']

            ratio = lambda_now / lambda_before
            old_threshold = master_node.threshold
            new_threshold = master_node.threshold * ratio

            master_node.threshold = new_threshold
            master_node._config['threshold'] = new_threshold
            logging.debug(f"Lambda changed from {lambda_before} to {lambda_now}. Threshold adapted from {old_threshold} to {new_threshold}.")

            lambda_before = lambda_now

        if master_node.finished_jobs == master_node._config['total_nb_jobs'] and len(master_node.waiting_jobs) == 0 and  len(master_node.all_jobs) == master_node._config['total_nb_jobs'] and len(master_node.tracker.ongoing_tasks) == 0 and len(master_node.queue.items) == 0:
            finished = True
            for compute_node in master_node.compute_nodes:
                if len(compute_node.queue.items) > 0:
                    finished = False
            if finished:
                break

def adaptationThetaV2(master_node):
    """
        this function adapte the threshold of all running jobs based of the lambda variability
        if lambda increase the threshold also increase with the same ratio
        it did the same for decreasing
    """
    lambda_before = master_node._config['lambda_rate']
    while True:

        yield master_node.env.timeout(0.01)
        stats = getStats(master_node)
        if lambda_before != master_node._config['lambda_rate']:
            print("lambda diff")
            lambda_now = master_node._config['lambda_rate']

            ratio = lambda_now / lambda_before
            old_threshold = master_node.threshold

            new_threshold = master_node.threshold * ratio  #need to add

            master_node.threshold = new_threshold
            master_node._config['threshold'] = new_threshold

            logging.debug(f"Lambda changed from {lambda_before} to {lambda_now}. Threshold adapted from {old_threshold} to {new_threshold}.")

            lambda_before = lambda_now

        if master_node.finished_jobs == master_node._config['total_nb_jobs'] and len(master_node.waiting_jobs) == 0 and  len(master_node.all_jobs) == master_node._config['total_nb_jobs'] and len(master_node.tracker.ongoing_tasks) == 0 and len(master_node.queue.items) == 0:
            finished = True
            for compute_node in master_node.compute_nodes:
                if len(compute_node.queue.items) > 0:
                    finished = False
            if finished:
                break

def getStats(master_node):
    """
        this function return the stats of the master node
    """
    nb_tasks_all = []
    dataset_sizes = []
    task_durations = []
    nb_jobs = 0

    # Job history
    for old_job in master_node.tracker.stats_on_jobs:
        nb_tasks_all.append(old_job["nb_tasks"])
        dataset_sizes.append(old_job["dataset size"])
        task_durations.append(old_job["task_execution_time"])
        nb_jobs += 1

    # Currently running jobs (excluding the target job)
    for running_job in master_node.running_jobs:
        nb_tasks_all.append(running_job.nb_tasks)
        dataset_sizes.append(running_job.dataset_size)
        if running_job.tasks and running_job.tasks[0].duration is not None:
            task_durations.append(running_job.tasks[0].duration)
        nb_jobs += 1

    stats = {
        "total_nb_jobs": master_node._config['total_nb_jobs'],
        "finished_jobs": master_node.finished_jobs,
        "waiting_jobs": len(master_node.waiting_jobs),
        "running_jobs": len(master_node.running_jobs),
        "all_jobs": len(master_node.all_jobs),
        "ongoing_tasks": len(master_node.tracker.ongoing_tasks),
        "queue_items": len(master_node.queue.items),
        "executed_job":len(master_node.tracker.stats_on_jobs),
        "tasks": nb_tasks_all,
        "dataset_sizes": dataset_sizes,
        "durations": task_durations,
    }

    return stats

def adaptationThetaV3(master_node, alpha=0.01, window_size=20):
    """
    Adapts theta based on:
    - the evolution of lambda (system load),
    - the average size of the last N jobs (number of tasks).
    => Even if lambda doesn't change, theta can move based on job size.
    """
    lambda_before = master_node._config['lambda_rate']
    while True:

        yield master_node.env.timeout(5)
        stats = getStats(master_node)

        lambda_now = master_node._config['lambda_rate']
        ratio = lambda_now / lambda_before if lambda_before > 0  else 1.0

        nb_free_nodes = len(master_node.select_availables_nodes(k=-1))

        old_threshold = master_node.threshold


        # average size over the last N jobs
        tasks_history = stats["tasks"][-window_size:]
        if len(tasks_history) > 0:
            avg_tasks = sum(tasks_history) / len(tasks_history)
        else:
            avg_tasks = 1  # avoid division by zero

        # correction factor based on average size
        size_factor = 1 / avg_tasks

        # new theta = depends on lambda + job size
        new_threshold = master_node.threshold * ratio * size_factor

        # user-defined bounds
        new_threshold = max(
            0.1,
            min(new_threshold, master_node._config['max_theta_user_preferences'])
        )

        master_node.threshold = new_threshold
        master_node._config['threshold'] = new_threshold

        logging.debug(
            f"[ADAPTATION] λ {lambda_before:.2f} → {lambda_now:.2f}, "
            f"avg_size({len(tasks_history)})={avg_tasks:.1f}, "
            f"θ {old_threshold:.3f} → {new_threshold:.3f} "
            f"(ratio={ratio:.3f}, size_factor={size_factor:.3f}) "
            f"threshold={old_threshold} → {master_node._config['threshold']}"
        )

        # update lambda_before for the next cycle
        lambda_before = lambda_now

        # stop condition
        if (master_node.finished_jobs == master_node._config['total_nb_jobs']
            and len(master_node.waiting_jobs) == 0
            and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
            and len(master_node.tracker.ongoing_tasks) == 0
            and len(master_node.queue.items) == 0):

            finished = True
            for compute_node in master_node.compute_nodes:
                if len(compute_node.queue.items) > 0:
                    finished = False
            if finished:
                break

def adaptationThetaV9_0(master_node, alpha=1, window_size=20,
                      w_load=1.0, w_size=1.0, w_resource=1.0,
                      sensitivity=1.0):
    """
    Works very well under load ✊🏻
    Dynamically adapts theta based on:
      - Load (lambda): less load => theta ↑
      - Average job size: small jobs => theta ↑
      - Available resources: more free nodes => theta ↑
    And keeps theta within [0.1, max_theta_user_preferences].
    The `sensitivity` parameter tunes how reactive it is.
    """
    lambda_ref = 10
    theta_ref = 1
    while True:
        yield master_node.env.timeout(10)
        stats = getStats(master_node)

        # Available resources
        nb_free_nodes = len(master_node.select_availables_nodes(k=-1))
        total_nodes = len(master_node.compute_nodes)
        free_ratio = nb_free_nodes / total_nodes if total_nodes > 0 else 0

        # Load and average size
        lambda_now = master_node._config['lambda_rate']
        tasks_history = stats["tasks"][-window_size:]
        avg_tasks = sum(tasks_history) / len(tasks_history) if tasks_history else 1

        # --- Recalibrated factors ---
        # 1. Low load => factor > 1
        load_factor = 1 - ((1/lambda_now) / ((1/lambda_now) + 1))
        # 2. Small jobs => factor > 1
        size_factor = 1 + (1 / (1 + alpha * avg_tasks))
        # 3. Plenty of resources => factor > 1
        resource_factor = 0.5 + free_ratio  # roughly 0.5 -> 1.5

        # --- Weighted combination ---
        combined_factor = (
            (load_factor ** w_load) *
            (size_factor ** w_size) *
            (resource_factor ** w_resource)
        )

        # --- Application and bounds ---
        base_theta = master_node._config.get('base_theta', 1.0)
        old_threshold = master_node.threshold

        new_threshold = base_theta * (combined_factor ** sensitivity)
        new_threshold = max(
            0.1,
            min(new_threshold, master_node._config['max_theta_user_preferences'])
        )

        master_node.threshold = new_threshold
        master_node._config['threshold'] = new_threshold

        master_node.tracker.log_threshold(new_threshold)

        logging.debug(
            f"[ADAPTATION] λ={lambda_now:.2f}, free={nb_free_nodes}/{total_nodes}, "
            f"avg_tasks={avg_tasks:.1f}, θ {old_threshold:.3f}→{new_threshold:.3f} | "
            f"load={load_factor:.3f}, size={size_factor:.3f}, res={resource_factor:.3f}"
        )

        # --- Stop condition ---
        if (master_node.finished_jobs == master_node._config['total_nb_jobs']
            and len(master_node.waiting_jobs) == 0
            and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
            and len(master_node.tracker.ongoing_tasks) == 0
            and len(master_node.queue.items) == 0):

            finished = all(len(n.queue.items) == 0 for n in master_node.compute_nodes)
            if finished:
                break


def adaptAccelerationThreshold(master_node, interval=None, sensitivity=None,
                                min_threshold=None, max_threshold=None):
    """
    Dynamically adapts master_node._config['acceleration_threshold'] - the threshold
    ACTUALLY used by checkUtilityProblem/SelectionnerMeilleurNoeud to decide whether to
    add a replica (unlike adaptationThetaV9_0 above, which adapts
    master_node.threshold/_config['threshold'], never read by this pipeline).

    Load signals, measured in real time (not a static value like lambda_rate):
      - waiting queue (waiting_jobs) relative to the number of nodes -> overload
      - ratio of free nodes (select_availables_nodes) -> available resources

    The more loaded the system is (long queue, few free nodes), the higher the threshold
    goes: in our formula (threshold = improvement / acceleration_threshold), a larger
    acceleration_threshold makes the test STRICTER (fewer replicas added, resources are
    preserved to place new jobs). Conversely, when the system is calm, the threshold
    drops back toward its base (permissive) value.

    `sensitivity` tunes reactivity (0 = never move, 1 = full amplitude between
    min_threshold and max_threshold).

    `adaptive_use_trend` (config, default False): instead of driving the threshold off
    the INSTANTANEOUS load signal (resource_factor/queue_factor of the current tick),
    fits a linear regression over the last `adaptive_trend_window` samples (time,
    signal) and uses the slope to PROJECT the signal `adaptive_trend_lookahead` seconds
    ahead.

    Two fixes turned out to be necessary for this to actually do anything useful
    (verified empirically on a bursty load, 20 nodes):

    1. The signal fed into the regression must NEVER be capped at 1.0 the way
       queue_factor=min(1.0,...) does for the instantaneous mode: under heavy overload
       (observed in practice: a queue of 100-300 jobs over 20 nodes, 0 free nodes
       continuously for >20000s), the queue DOES genuinely drop (312 -> 98) but stays
       far above total_nodes throughout, so the capped ratio stays stuck at exactly 1.0
       the whole time - the regression sees a perfectly flat line (zero slope) and can't
       anticipate anything, no matter the window size.

    2. Even uncapped, a raw ratio (queue_depth/total_nodes, e.g. 15.6) stays unbounded
       and, more importantly, off-scale relative to the useful [0,1] window: a
       regression can well detect that it's dropping from 15.6 to 8.3, but at that rate
       it would still take ~16000s to get back under 1.0 - a reasonable projection
       horizon (hundreds to a few thousand seconds) isn't enough to close that scale
       gap. _compress_ratio (x/(x+scale)) replaces the hard cap with a soft
       compression, bounded within [0,1) but still SENSITIVE to variations even well
       above 1 (unlike min(1.0,x), which flattens everything above the threshold to a
       strictly identical value): going from x=15 to x=8 at scale=1 gives
       0.938 -> 0.889, real movement, whereas the old cap gave 1.0 -> 1.0 in both cases.

    Both fixes only apply to the adaptive_use_trend=True branch - the default
    instantaneous mode keeps EXACTLY the old formula (already validated: ~4.7% flow-time
    gain, ~28% fewer replicas under load), so there is no regression on what already
    worked. With fewer than 2 samples in the window (very start of the simulation), the
    trend branch falls back to the current signal - identical behavior to
    adaptive_use_trend=False at that specific moment.
    """
    base_threshold = master_node._config['acceleration_threshold']
    min_threshold = min_threshold if min_threshold is not None else base_threshold
    max_threshold = (
        max_threshold if max_threshold is not None
        else master_node._config.get('max_acceleration_threshold', base_threshold * 40)
    )
    interval = interval if interval is not None else master_node._config.get('adaptive_interval', 10)
    sensitivity = sensitivity if sensitivity is not None else master_node._config.get('adaptive_sensitivity', 1.0)

    use_trend = master_node._config.get('adaptive_use_trend', False)
    trend_window = master_node._config.get('adaptive_trend_window', 6)
    trend_lookahead = master_node._config.get('adaptive_trend_lookahead', interval * 2)
    compression_scale = master_node._config.get('adaptive_queue_compression_scale', 1.0)
    history = []  # [(t, raw_signal), ...], capped at trend_window samples

    def _compress_ratio(x, scale):
        """[0, +inf) -> [0, 1): x=0 -> 0, x=scale -> 0.5, x -> infinity -> 1, derivative
        always > 0 (never flat like with min(1.0, x)) - see docstring above."""
        return x / (x + scale)

    while True:
        yield master_node.env.timeout(interval)

        total_nodes = len(master_node.compute_nodes)
        nb_free_nodes = len(master_node.select_availables_nodes(k=-1))
        free_ratio = nb_free_nodes / total_nodes if total_nodes > 0 else 1.0
        resource_factor = 1 - free_ratio  # 0 = everything free, 1 = nothing free

        queue_depth = len(master_node.waiting_jobs)
        queue_factor = min(1.0, queue_depth / total_nodes) if total_nodes > 0 else 0.0

        raw_signal = (resource_factor + queue_factor) / 2
        raw_signal_for_log = raw_signal  # raw INSTANTANEOUS signal, for comparison in plots

        if use_trend:
            # Soft compression instead of hard capping (see docstring): stays sensitive
            # to variations even well above total_nodes, unlike min(1.0, ratio), which
            # treats everything above the threshold as identical.
            queue_ratio = queue_depth / total_nodes if total_nodes > 0 else 0.0
            queue_factor_compressed = _compress_ratio(queue_ratio, compression_scale)
            raw_signal_compressed = (resource_factor + queue_factor_compressed) / 2
            raw_signal_for_log = raw_signal_compressed  # apples-to-apples comparison (same compression)

            history.append((master_node.env.now, raw_signal_compressed))
            if len(history) > trend_window:
                history.pop(0)
            if len(history) >= 2:
                times = np.array([t for t, _ in history])
                values = np.array([v for _, v in history])
                slope, intercept = np.polyfit(times, values, 1)
                trend_now = slope * master_node.env.now + intercept
                projected_signal = trend_now + slope * trend_lookahead
            else:
                projected_signal = raw_signal_compressed
            signal_for_threshold = max(0.0, min(1.0, projected_signal))
        else:
            signal_for_threshold = raw_signal

        overload_signal = sensitivity * signal_for_threshold
        overload_signal = max(0.0, min(1.0, overload_signal))

        new_threshold = min_threshold + (max_threshold - min_threshold) * overload_signal

        old_threshold = master_node._config['acceleration_threshold']
        master_node._config['acceleration_threshold'] = new_threshold
        master_node.tracker.log_threshold(new_threshold, signal=signal_for_threshold, raw_signal=raw_signal_for_log)

        logging.debug(
            f"[ADAPT acceleration_threshold] free={nb_free_nodes}/{total_nodes} "
            f"queue={queue_depth} overload={overload_signal:.2f} "
            f"threshold {old_threshold:.3f}->{new_threshold:.3f}"
        )

        if (master_node.finished_jobs == master_node._config['total_nb_jobs']
            and len(master_node.waiting_jobs) == 0
            and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
            and len(master_node.tracker.ongoing_tasks) == 0
            and len(master_node.queue.items) == 0):

            finished = all(len(n.queue.items) == 0 for n in master_node.compute_nodes)
            if finished:
                break


def learnAccelerationThresholdStartup(master_node, candidates=None, window_duration=None,
                                       recheck_interval=None, drift_tolerance=None, metric=None):
    """"Control engineering" auto-tuning of acceleration_threshold: instead of a fixed
    formula (adaptAccelerationThreshold above, whose "high load -> strict threshold"
    relationship turned out to be empirically FALSE on workloads-100-storage-contrainte -
    see xp_threshold_grid_search: mean_flow_time is flat/noisy for threshold in
    [0.01, 8] then degrades sharply beyond that, with the empirical optimum around 4, not
    16-50), we SACRIFICE an initial portion of the workload to exploration
    (explore-then-commit) to learn the right value for THIS SPECIFIC workload/infra,
    rather than guessing it once and for all.

    EXPLORE phase: time is split into windows of `window_duration`. One candidate from
    `candidates` is active per window (round-robin, one full pass = len(candidates)
    windows). At the end of each window, we measure the average flow_time of jobs that
    FINISHED during that window (an imperfect proxy - a job may have lived under several
    candidates - but good enough to spot the elbow, like a relay-feedback test in control
    engineering that isn't after the exact curve, just the switching point).

    COMMIT phase: once every candidate has been tried at least once (with at least one
    finished job), acceleration_threshold is fixed to the best candidate observed
    (lowest average flow_time) and that value is kept.

    Post-commit monitoring: every `recheck_interval` seconds, the recent average
    flow_time (jobs finished since the last recheck) is compared to the baseline
    recorded for the candidate chosen during exploration. If it has drifted by more than
    `drift_tolerance` (relative, worse direction), a short re-exploration is triggered
    (a new pass over `candidates`) and we re-commit to the best one - "adapt along the
    way" if conditions change, without continuously re-exploring while everything is
    fine.
    """
    candidates = candidates if candidates is not None else master_node._config.get(
        'threshold_learning_candidates', [0.05, 0.5, 2, 8, 16])
    window_duration = window_duration if window_duration is not None else master_node._config.get(
        'threshold_learning_window', 200)
    recheck_interval = recheck_interval if recheck_interval is not None else master_node._config.get(
        'threshold_learning_recheck_interval', window_duration * len(candidates) * 3)
    drift_tolerance = drift_tolerance if drift_tolerance is not None else master_node._config.get(
        'threshold_learning_drift_tolerance', 0.25)
    metric = metric if metric is not None else master_node._config.get('threshold_learning_metric', 'flow_time')
    pressure_gate = master_node._config.get('threshold_learning_pressure_gate', False)
    pressure_wait_threshold = master_node._config.get('threshold_learning_pressure_wait_threshold', 1.0)
    pressure_queue_threshold = master_node._config.get('threshold_learning_pressure_queue_threshold', 1)

    base_threshold = master_node._config['acceleration_threshold']
    history = []  # (env.now, theta, phase) - phase in {"gate_wait", "explore", "commit"}

    def _finished():
        return (master_node.finished_jobs == master_node._config['total_nb_jobs']
                and len(master_node.waiting_jobs) == 0
                and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
                and len(master_node.tracker.ongoing_tasks) == 0
                and len(master_node.queue.items) == 0
                and all(len(n.queue.items) == 0 for n in master_node.compute_nodes))

    def _has_pressure(since_index):
        """Real contention signal: a non-empty queue RIGHT NOW, or an already-significant
        waiting_time on freshly finished jobs - NOT just a flow_time that moves (it can
        vary with job size without any contention at all). Without a genuine pressure
        signal, exploring only ever PAYS the cost (strict candidates like 8/16 that
        degrade things) without ever being able to earn it back - observed empirically at
        lambda=100 (no contention at all), where migration+startup_learning lost up to
        +82% vs. no migration at all."""
        new_jobs = master_node.tracker.stats_on_jobs[since_index:]
        recent_wait = (
            float(np.mean([j['starting_time'] - j['arriving_time'] for j in new_jobs]))
            if new_jobs else 0.0
        )
        return recent_wait > pressure_wait_threshold or len(master_node.waiting_jobs) >= pressure_queue_threshold

    def _set(value, phase="explore"):
        master_node._config['acceleration_threshold'] = value
        history.append((master_node.env.now, value, phase))

    def _mean_flow_time_since(stats_on_jobs, since_index):
        """Average cost of stats_on_jobs[since_index:] (already finished at this point) -
        according to `metric`: flow_time (finishing-arriving, default) or waiting_time
        (starting-arriving)."""
        new_jobs = stats_on_jobs[since_index:]
        if not new_jobs:
            return None
        if metric == 'waiting_time':
            values = [j['starting_time'] - j['arriving_time'] for j in new_jobs]
        else:
            values = [j['finishing_time'] - j['arriving_time'] for j in new_jobs]
        return float(np.mean(values))

    def _explore_pass(committed_stats):
        """One full pass over `candidates`, one window each. Returns the best candidate
        observed (lowest average flow_time among those that saw at least one finished
        job) and updates committed_stats[candidate] = observed average flow_time."""
        best_candidate, best_flow = None, float('inf')
        for candidate in candidates:
            _set(candidate)
            n_before = len(master_node.tracker.stats_on_jobs)
            yield master_node.env.timeout(window_duration)
            mean_flow = _mean_flow_time_since(master_node.tracker.stats_on_jobs, n_before)
            if mean_flow is not None:
                committed_stats[candidate] = mean_flow
                logging.debug(
                    f"[THRESHOLD-LEARN] explore candidate={candidate} "
                    f"mean_flow={mean_flow:.1f} (n={len(master_node.tracker.stats_on_jobs) - n_before})"
                )
                if mean_flow < best_flow:
                    best_candidate, best_flow = candidate, mean_flow
        return best_candidate, best_flow

    if pressure_gate:
        while True:
            n_before = len(master_node.tracker.stats_on_jobs)
            yield master_node.env.timeout(window_duration)
            if _finished():
                master_node.tracker.threshold_learning_history = history
                return
            if _has_pressure(n_before):
                logging.info(f"[THRESHOLD-LEARN] pressure detected at t={master_node.env.now:.0f} "
                             f"- starting exploration")
                break
            history.append((master_node.env.now, base_threshold, "gate_wait"))

    committed_stats = {}
    best_candidate, best_flow = yield from _explore_pass(committed_stats)

    if best_candidate is None:
        # No job finished during exploration (workload too short / load too low): fall
        # back to the base value instead of staying on the last candidate tried at
        # random.
        _set(base_threshold, phase="commit")
        best_candidate, best_flow = base_threshold, None
    else:
        _set(best_candidate, phase="commit")

    logging.info(f"[THRESHOLD-LEARN] committing to acceleration_threshold={best_candidate} "
                 f"(exploration mean_flow={best_flow})")

    while True:
        n_before = len(master_node.tracker.stats_on_jobs)
        yield master_node.env.timeout(recheck_interval)

        if (master_node.finished_jobs == master_node._config['total_nb_jobs']
            and len(master_node.waiting_jobs) == 0
            and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
            and len(master_node.tracker.ongoing_tasks) == 0
            and len(master_node.queue.items) == 0
            and all(len(n.queue.items) == 0 for n in master_node.compute_nodes)):
            break

        recent_flow = _mean_flow_time_since(master_node.tracker.stats_on_jobs, n_before)
        if recent_flow is None or best_flow is None:
            continue

        if recent_flow > best_flow * (1 + drift_tolerance) and (not pressure_gate or _has_pressure(n_before)):
            logging.info(f"[THRESHOLD-LEARN] drift detected (recent={recent_flow:.1f} vs "
                         f"baseline={best_flow:.1f}) - re-exploring")
            best_candidate, best_flow = yield from _explore_pass(committed_stats)
            if best_candidate is None:
                _set(base_threshold, phase="commit")
                best_candidate, best_flow = base_threshold, None
            else:
                _set(best_candidate, phase="commit")
            logging.info(f"[THRESHOLD-LEARN] re-committing to acceleration_threshold={best_candidate} "
                         f"(mean_flow={best_flow})")

    master_node.tracker.threshold_learning_history = history


def hybridStartupPO(master_node, candidates=None, window_duration=None, metric=None,
                     po_alpha_fraction=None, **po_kwargs):
    """Hybrid approach requested by the user: startup_learning quickly finds the right
    ZONE (a coarse grid of spaced-out candidates, e.g. [0.05, 0.5, 2, 8, 16] - robust to
    noise but coarse resolution, can never return a value outside its list, e.g. never
    exactly 4.0 or 0.2 even if that's the true optimum), then a CONTINUOUS P&O refines
    around that starting point (fine resolution, but fragile on its own if started from
    the default 0.05 - see every P&O run from this session that never reached theta*=4.0
    when starting from 0.05).

    Phase A (a single pass, no infinite monitoring loop like in
    learnAccelerationThresholdStartup): tests each candidate for one window, keeps the
    best one observed.

    Phase B: launches controlLoopAccelerationThreshold (detrended P&O, see its
    docstring) starting from theta = the chosen candidate, with an alpha_initial
    PROPORTIONAL to that starting point (`po_alpha_fraction` * theta, default 15%)
    rather than the usual fixed value (0.3) - a fixed step would mean something very
    different depending on whether we start from 0.05 or 8. Phase B's regime detector
    (local 3-point re-exploration) is enough to react to a load change along the way; no
    need to duplicate learnAccelerationThresholdStartup's drift monitoring.
    """
    candidates = candidates if candidates is not None else master_node._config.get(
        'threshold_learning_candidates', [0.05, 0.5, 2, 8, 16])
    window_duration = window_duration if window_duration is not None else master_node._config.get(
        'threshold_learning_window', 200)
    metric = metric if metric is not None else master_node._config.get('threshold_learning_metric', 'flow_time')
    po_alpha_fraction = po_alpha_fraction if po_alpha_fraction is not None else master_node._config.get(
        'threshold_hybrid_po_alpha_fraction', 0.15)

    base_threshold = master_node._config['acceleration_threshold']

    def _mean_cost_since(since_index):
        new_jobs = master_node.tracker.stats_on_jobs[since_index:]
        if not new_jobs:
            return None
        if metric == 'waiting_time':
            values = [j['starting_time'] - j['arriving_time'] for j in new_jobs]
        else:
            values = [j['finishing_time'] - j['arriving_time'] for j in new_jobs]
        return float(np.mean(values))

    # --- Phase A: explore-then-commit, A SINGLE pass ---
    best_candidate, best_cost = None, float('inf')
    for candidate in candidates:
        master_node._config['acceleration_threshold'] = candidate
        n_before = len(master_node.tracker.stats_on_jobs)
        yield master_node.env.timeout(window_duration)
        cost = _mean_cost_since(n_before)
        if cost is not None:
            logging.debug(f"[HYBRID] explore candidate={candidate} cost={cost:.1f}")
            if cost < best_cost:
                best_candidate, best_cost = candidate, cost

    theta_start = best_candidate if best_candidate is not None else base_threshold
    master_node._config['acceleration_threshold'] = theta_start
    logging.info(f"[HYBRID] Phase A done - committing to theta_start={theta_start} "
                 f"(exploration cost={best_cost if best_candidate is not None else None}), "
                 f"switching to fine-grained P&O (alpha_initial={po_alpha_fraction * theta_start:.4f})")

    # --- Phase B: fine-grained P&O around theta_start ---
    yield from controlLoopAccelerationThreshold(
        master_node,
        alpha_initial=max(po_alpha_fraction * theta_start, master_node._config.get('threshold_control_alpha_min', 0.02)),
        metric=metric,
        **po_kwargs,
    )


def controlLoopAccelerationThreshold(master_node, window_size=None, alpha_initial=None, alpha_min=None,
                                      alpha_max=None, gamma_up=None, gamma_down=None,
                                      min_threshold=None, max_threshold=None,
                                      epsilon_abs=None, epsilon_rel=None,
                                      trend_window=None, initial_direction=None,
                                      dead_zone_freeze=None, use_detrending=None, detrend_signal_mode=None,
                                      no_pressure_threshold=None, no_pressure_min_windows=None,
                                      regime_detection=None, regime_window=None, regime_multiplier=None,
                                      metric=None):
    """Perturb & Observe (extremum-seeking) for acceleration_threshold - 3rd iteration in
    this session, after diagnosing 3 distinct bugs in the 2 previous ones (reported to
    the user in detail):
      1. A dead zone that "confirmed and accelerated" instead of "freezing" -> ran away
         to the ceiling whenever there was no signal (low load, waiting_time~0
         everywhere).
      2. A CUSUM drift detector that fired on EVERY window because the raw signal
         (flow_time/waiting_time) grows organically under sustained load (queue backlog
         building up, nothing to do with theta) - the "slow baseline" never caught up
         with that growth, so it forced permanent oscillation between 2 values.
      3. Even without CUSUM, the raw P&O rule (dJ = J_k - J_{k-1}) is fooled by that same
         organic trend: dJ>0 almost all the time, so direction reversal is nearly
         systematic, never a real exploration toward a distant optimum (theta*=4 stays
         out of reach, the controller oscillates around 0.25-0.30).

    This version fixes all 3:

    (a) DEAD ZONE = FREEZE, not confirmation. Insufficient signal (|signal| < epsilon)
        -> theta DOES NOT MOVE, direction=0, alpha reset to alpha_initial (no "bold
        driver" growth on noise).

    (b) DETRENDING: instead of comparing raw J_k to J_{k-1}, the local trend is
        estimated (linear regression over the last `trend_window` windows, an
        OUT-OF-SAMPLE prediction - fit on J_{k-m..k-1}, extrapolated to index k BEFORE
        observing J_k) and the decision is based on the RESIDUAL
        residual_k = J_k - Jhat_k, which isolates theta's effect from the system's
        natural drift (queue backlog, etc.) - see model J(theta,t) = T(t) + G(theta,t) +
        noise, we only want to react to G, not to T. `detrend_signal_mode`: "residual"
        (raw residual_k) or "diff_residual" (residual_k - residual_{k-1}) - both are
        exposed, to compare experimentally (see xp_detrended_po_comparison.py).

    (c) "NO PRESSURE" SAFEGUARD: if waiting_time stays ~zero over the last
        `no_pressure_min_windows` windows (`no_pressure_threshold`), the controller is
        explicitly frozen (nothing to optimize) instead of leaving a near-zero dJ/residual
        in the old "confirm by default" dead zone.

    RELATIVE epsilon (not a single absolute threshold): epsilon = epsilon_abs +
    epsilon_rel*|J_k| - stays relevant regardless of flow_time's order of magnitude
    (tens vs. thousands of seconds depending on the instance).

    Regime-change detection (optional, `regime_detection`): much more cautious than the
    previous CUSUM - based on the RESIDUAL (already detrended, so normally close to
    white noise in a stable regime) rather than the raw signal. If the mean of the last
    `regime_window` residuals deviates by more than `regime_multiplier` times the usual
    noise scale (slow EWMA of |residual|), a genuine regime/workload change is assumed
    (not just organic drift, already removed by detrending) and a short exploration
    burst is triggered (3 candidates spread around theta) before resuming the detrended
    P&O.

    Decision rule (theta_{k+1} = clip(theta_k + direction*alpha, min, max)):
        |signal| < epsilon                  -> FREEZE (direction=0, alpha=alpha_initial, theta unchanged)
        signal < -epsilon (improvement)     -> keep/initialize direction, alpha *= gamma_up (capped)
        signal > +epsilon (degradation)     -> reverse direction, alpha *= gamma_down (floored)

    J_k = mean_flow_time over a window of `window_size` jobs (NOT waiting_time - unlike
    the previous version, per explicit request: the reference pseudo-code specifies
    flow_time as the cost to minimize, and detrending is meant to remove the bias that
    waiting_time was meant to avoid by another means).
    """
    window_size = window_size if window_size is not None else master_node._config.get('threshold_control_window_size', 5)
    alpha_initial = alpha_initial if alpha_initial is not None else master_node._config.get('threshold_control_alpha0', 0.3)
    alpha_min = alpha_min if alpha_min is not None else master_node._config.get('threshold_control_alpha_min', 0.02)
    alpha_max = alpha_max if alpha_max is not None else master_node._config.get('threshold_control_alpha_max', 1.5)
    gamma_up = gamma_up if gamma_up is not None else master_node._config.get('threshold_control_gamma_up', 1.12)
    gamma_down = gamma_down if gamma_down is not None else master_node._config.get('threshold_control_gamma_down', 0.5)
    min_threshold = min_threshold if min_threshold is not None else master_node._config.get('threshold_control_min', 0.01)
    max_threshold = max_threshold if max_threshold is not None else master_node._config.get(
        'threshold_control_max', master_node._config.get('max_acceleration_threshold', 16.0))
    epsilon_abs = epsilon_abs if epsilon_abs is not None else master_node._config.get('threshold_control_epsilon_abs', 1.0)
    epsilon_rel = epsilon_rel if epsilon_rel is not None else master_node._config.get('threshold_control_epsilon_rel', 0.05)
    trend_window = trend_window if trend_window is not None else master_node._config.get('threshold_control_trend_window', 5)
    initial_direction = initial_direction if initial_direction is not None else master_node._config.get(
        'threshold_control_initial_direction', 1)
    dead_zone_freeze = dead_zone_freeze if dead_zone_freeze is not None else master_node._config.get(
        'threshold_control_dead_zone_freeze', True)
    use_detrending = use_detrending if use_detrending is not None else master_node._config.get(
        'threshold_control_use_detrending', True)
    detrend_signal_mode = detrend_signal_mode if detrend_signal_mode is not None else master_node._config.get(
        'threshold_control_detrend_signal_mode', 'residual')
    no_pressure_threshold = no_pressure_threshold if no_pressure_threshold is not None else master_node._config.get(
        'threshold_control_no_pressure_threshold', 1.0)
    no_pressure_min_windows = no_pressure_min_windows if no_pressure_min_windows is not None else master_node._config.get(
        'threshold_control_no_pressure_min_windows', 3)
    regime_detection = regime_detection if regime_detection is not None else master_node._config.get(
        'threshold_control_regime_detection', True)
    regime_window = regime_window if regime_window is not None else master_node._config.get('threshold_control_regime_window', 8)
    regime_multiplier = regime_multiplier if regime_multiplier is not None else master_node._config.get(
        'threshold_control_regime_multiplier', 4.0)
    metric = metric if metric is not None else master_node._config.get('threshold_control_metric', 'flow_time')

    def _finished():
        return (master_node.finished_jobs == master_node._config['total_nb_jobs']
                and len(master_node.waiting_jobs) == 0
                and len(master_node.all_jobs) == master_node._config['total_nb_jobs']
                and len(master_node.tracker.ongoing_tasks) == 0
                and len(master_node.queue.items) == 0
                and all(len(n.queue.items) == 0 for n in master_node.compute_nodes))

    def _trimmed_mean(values, trim=0.1):
        values = sorted(values)
        k = int(len(values) * trim)
        core = values[k: len(values) - k] if len(values) - 2 * k > 0 else values
        return float(np.mean(core))

    def _fit_trend_and_predict(history, k_index):
        """Linear regression over `history` (list of J, consecutive indices ending
        right before k_index), extrapolated to k_index - an OUT-OF-SAMPLE prediction
        (uses no information about J_k itself)."""
        m = len(history)
        idx = np.arange(k_index - m, k_index)
        b, a = np.polyfit(idx, history, 1)  # J ~ a + b*i
        return a + b * k_index

    theta = master_node._config['acceleration_threshold']
    alpha = alpha_initial
    direction = 0  # neutral at the start - initial_direction is only taken on the first real move
    J_history = []          # raw J_k values, used to fit the trend
    residual_history = []   # residuals (or raw dJ if use_detrending=False), used by the regime detector
    residual_scale_ewma = None
    recent_waits = []
    n_seen = 0
    k = 0  # window index (for the regression over consecutive indices)
    history_log = []  # exposed for analysis/plots: (env.now, k, theta, J_k, Jhat_k, residual, signal, direction, alpha)

    def _reexplore_burst(candidates):
        """Short re-exploration burst (3 points spread around the current theta) -
        reuses the explore-then-commit spirit of learnAccelerationThresholdStartup but
        capped at 3 windows, triggered only by the regime detector (rare)."""
        nonlocal theta
        best_c, best_j = theta, float('inf')
        for c in candidates:
            c = min(max_threshold, max(min_threshold, c))
            master_node._config['acceleration_threshold'] = c
            n_before = len(master_node.tracker.stats_on_jobs)
            while len(master_node.tracker.stats_on_jobs) - n_before < window_size:
                yield master_node.env.timeout(1)
                if _finished():
                    return
            flows = [j['finishing_time'] - j['arriving_time']
                     for j in master_node.tracker.stats_on_jobs[n_before:n_before + window_size]]
            j_c = _trimmed_mean(flows)
            if j_c < best_j:
                best_c, best_j = c, j_c
        theta = best_c
        master_node._config['acceleration_threshold'] = theta

    while True:
        yield master_node.env.timeout(1)  # fine granularity - act as soon as a window is ready

        if _finished():
            break

        all_jobs = master_node.tracker.stats_on_jobs
        while len(all_jobs) - n_seen >= window_size:
            window = all_jobs[n_seen: n_seen + window_size]
            n_seen += window_size
            k += 1

            waits = [j['starting_time'] - j['arriving_time'] for j in window]
            if metric == 'waiting_time':
                J_k = _trimmed_mean(waits)
            else:
                flow_times = [j['finishing_time'] - j['arriving_time'] for j in window]
                J_k = _trimmed_mean(flow_times)
            wait_k = _trimmed_mean(waits)

            recent_waits.append(wait_k)
            recent_waits = recent_waits[-no_pressure_min_windows:]

            # --- "no pressure" safeguard: nothing to optimize, don't explore blindly
            # (avoids the runaway observed at lambda=100: theta climbed to the ceiling
            # even though mean_wait was already 0 everywhere, for lack of a signal to
            # stop it). ---
            if len(recent_waits) >= no_pressure_min_windows and max(recent_waits) < no_pressure_threshold:
                direction = 0
                alpha = alpha_initial
                J_history.append(J_k)
                J_history = J_history[-(trend_window + regime_window + 2):]
                history_log.append((master_node.env.now, k, theta, J_k, None, None, direction, alpha))
                continue

            J_history.append(J_k)
            J_history = J_history[-(trend_window + regime_window + 2):]

            if use_detrending:
                if len(J_history) < trend_window + 1:
                    history_log.append((master_node.env.now, k, theta, J_k, None, None, direction, alpha))
                    continue
                fit_history = J_history[-(trend_window + 1):-1]
                Jhat_k = _fit_trend_and_predict(fit_history, k)
                residual = J_k - Jhat_k
            else:
                Jhat_k = J_history[-2] if len(J_history) >= 2 else None
                if Jhat_k is None:
                    history_log.append((master_node.env.now, k, theta, J_k, None, None, direction, alpha))
                    continue
                residual = J_k - Jhat_k

            residual_history.append(residual)
            residual_history = residual_history[-(regime_window + 1):]
            residual_scale_ewma = (
                abs(residual) if residual_scale_ewma is None
                else 0.1 * abs(residual) + 0.9 * residual_scale_ewma
            )

            if detrend_signal_mode == 'diff_residual' and len(residual_history) >= 2:
                signal = residual_history[-1] - residual_history[-2]
            else:
                signal = residual

            # --- regime-change detection: on the RESIDUAL (already detrended), not on
            # the raw signal - much less prone to false alarms than the previous CUSUM,
            # since normal organic drift has already been removed by this point. ---
            if (regime_detection and len(residual_history) >= regime_window
                    and residual_scale_ewma and residual_scale_ewma > 1e-6):
                recent_mean = float(np.mean(residual_history[-regime_window:]))
                if abs(recent_mean) > regime_multiplier * residual_scale_ewma:
                    logging.info(f"[THRESHOLD-P&O] regime change detected "
                                 f"(mean residual={recent_mean:.2f}, scale={residual_scale_ewma:.2f}) "
                                 f"- re-exploration burst")
                    yield from _reexplore_burst([theta / 3, theta, theta * 3])
                    J_history, residual_history, residual_scale_ewma = [], [], None
                    direction, alpha = 0, alpha_initial
                    continue

            epsilon = epsilon_abs + epsilon_rel * abs(J_k)

            if abs(signal) < epsilon:
                if dead_zone_freeze:
                    direction = 0
                    alpha = alpha_initial
                    # theta unchanged
                else:
                    # historical (buggy) behavior, kept for controlled comparison via
                    # threshold_control_dead_zone_freeze=False: confirms the direction
                    # and lets alpha keep growing even without a reliable signal.
                    if direction == 0:
                        direction = initial_direction
                    alpha = min(alpha_max, gamma_up * alpha)
                    theta = min(max_threshold, max(min_threshold, theta + direction * alpha))
                    master_node._config['acceleration_threshold'] = theta
            elif signal < 0:  # improvement: observed cost below what the trend predicted
                if direction == 0:
                    direction = initial_direction
                alpha = min(alpha_max, gamma_up * alpha)
                theta = min(max_threshold, max(min_threshold, theta + direction * alpha))
                master_node._config['acceleration_threshold'] = theta
            else:  # degradation: observed cost above the predicted trend
                direction = -direction if direction != 0 else -initial_direction
                alpha = max(alpha_min, gamma_down * alpha)
                theta = min(max_threshold, max(min_threshold, theta + direction * alpha))
                master_node._config['acceleration_threshold'] = theta

            master_node.tracker.log_threshold(theta, signal=signal, raw_signal=J_k)
            history_log.append((master_node.env.now, k, theta, J_k, Jhat_k, residual, direction, alpha))

            logging.debug(
                f"[THRESHOLD-P&O] J_k={J_k:.2f} Jhat_k={Jhat_k if Jhat_k is None else round(Jhat_k,2)} "
                f"residual={residual:.2f} signal={signal:.2f} eps={epsilon:.2f} "
                f"dir={direction:+d} alpha={alpha:.3f} theta->{theta:.3f}"
            )

    master_node.threshold_control_history = history_log
    master_node.tracker.threshold_control_history = history_log  # exposed through the tracker returned by utilityBasedReplicationHeterogeneousNodes (master_node itself isn't exposed to the caller)
