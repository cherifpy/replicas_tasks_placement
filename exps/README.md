# Running experiments — how the code works

This folder has one file, `run_experiment.py`, a single command-line entry point that
runs any of the 3 approaches evaluated in the paper ("Cost-Aware Replicas Placement and
Tasks Scheduling In Geo-Distributed Infrastructure", Si-Mohammed et al.) on a given
instance, with the arguments that matter for that approach.

It does not reimplement anything — see "How it works" below for why, and read each
method's own README (`../README.md`, `../offline-cop/README.md`, `../online-cop/README.md`)
for how that method's actual scheduling logic works.

## Quick start

```bash
cd simulator/paper-artifact/exps

# Generate the traces once, if traces/ is empty (see ../traces/generate_traces.py)
python3 ../traces/generate_traces.py

# Heuristic (Section 4.2)
python3 run_experiment.py heuristic inst-20J-50N --sigma 0.05

# Offline COP (Section 4.1, whole-horizon CP solve)
python3 run_experiment.py offline inst-20J-50N --time-limit 150s

# Online COP (Section 4.1, re-solved on every job arrival)
python3 run_experiment.py online inst-20J-50N --solver-time-limit 5
```

`inst-20J-50N` is looked up under `../traces/` automatically — you can also pass a full
path to any instance directory (e.g. one you generated yourself with different
parameters).

Each subcommand's own `--help` lists exactly what that method's arguments mean:
```bash
python3 run_experiment.py heuristic --help
python3 run_experiment.py offline --help
python3 run_experiment.py online --help
```

## Collecting results across runs

Every subcommand accepts `--json-out FILE`: the parsed final summary of that one run
(instance, method, `avg_flow_time_s`, `nb_transfers`, the method-specific parameters
used, wall-clock time...) is appended as a single JSON line to `FILE`. Run several
methods/instances/parameters against the same file to build a dataset for comparison:

```bash
for method in heuristic offline online; do
    python3 run_experiment.py "$method" inst-20J-50N --json-out results.jsonl
done
```

```python
import pandas as pd
df = pd.read_json("results.jsonl", lines=True)
df.groupby("method")["avg_flow_time_s"].mean()
```

`avg_flow_time_s` is the metric to compare across methods — all 3 runners compute it the
same way (mean of `finishing_time - arriving_time` over every finished job), regardless
of how differently each one arrives at its schedule.

## How it works

`run_experiment.py` is a thin **subprocess orchestrator**, not a shared library:

- `heuristic` → runs `python3 ../run_heuristic.py <instance> --sigma ... --until ...`
  from `../` (see `../README.md`).
- `offline` → compiles `../offline-cop/src/*.java` once (skipped on later runs unless
  `--recompile` is passed) and runs
  `java -cp "out:lib/*" RunOfflineCOP <instance> [output_dir]` from `../offline-cop/`,
  with `OFFLINE_COP_TIME_LIMIT` set from `--time-limit` (see `../offline-cop/README.md`).
- `online` → runs `python3 ../online-cop/run_online_cop.py <instance> --until ... --solver-time-limit ...`
  from `../online-cop/`, with `ONLINE_COP_OBJECTIVE` set from `--objective`
  (see `../online-cop/README.md`).

Each subprocess's own stdout is streamed live (so you still see the underlying tool's
own progress/debug output, e.g. the Choco solver's search log for `offline`/`online`),
and its final `key: value` summary block is parsed into a Python dict with a small
regex (`_RESULT_LINE_RE`) — the same "key: value" lines you'd see running that method by
hand, nothing solver/scheduler-specific is parsed.

### Why subprocesses instead of importing each method's code directly

The 3 approaches are **not just 3 functions in one shared codebase** — they are 3
independent, self-contained engines that happen to share a trace format:

- **Heuristic** (`../`) is plain Python + SimPy: a discrete-event simulation where a
  `Job` arrives, the `UtilityBasedHeterogeneousApproach` master decides which
  `ComputeNode`(s) to replicate its dataset to (via the utility/acceleration test), and
  each node processes its queued tasks.
- **Offline COP** (`../offline-cop/`) is pure Java: one Choco-solver constraint model
  covering the *entire* instance (every job, every node, the whole horizon) is built and
  solved once with Large Neighborhood Search. There is no simulation loop — the solver's
  output *is* the schedule.
- **Online COP** (`../online-cop/`) is Python + SimPy **driving** the same kind of
  Choco-solver model, but re-solved from scratch on every job arrival over only the
  jobs/tasks not yet started, fed a node-free-time *estimate* instead of the whole
  horizon (see `SchedulingUsingCSPOnline.nodesFreeTime` in
  `../online-cop/master_node_online.py`).

The Heuristic and Online COP therefore both need a `Job`/`ComputeNode`/`Tracker`
SimPy engine — but they use **two different, incompatible copies** of those classes
(`../job.py`/`../compute_node.py`/`../tracker.py` vs.
`../online-cop/job.py`/`../online-cop/compute_node.py`/`../online-cop/tracker.py` — see
"Different engines" in `../online-cop/README.md` for the concrete difference: one nodes
model holds a single resident dataset at a time, the other several; `compute_capacity`
is a divisor in one, a multiplier in the other). Importing both sets of same-named
modules into one Python process would collide in `sys.modules` — whichever is imported
first silently "wins" and the other approach would run with the wrong classes. Running
each method as its own subprocess (exactly as if you'd typed its command by hand) keeps
their module namespaces completely separate, at the cost of a little process-startup
overhead per run — negligible next to the solver time budgets involved.

### Instance resolution

`resolve_instance()` accepts either a bare name (`inst-20J-50N`, looked up under
`../traces/`) or any path to an instance directory, and always converts it to an
**absolute** path before building a subprocess command — each method is invoked from
its *own* directory (`../`, `../offline-cop/`, `../online-cop/`), so a relative path
typed by the user would otherwise be resolved against the wrong working directory.
