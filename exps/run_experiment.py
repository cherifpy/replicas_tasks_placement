#!/usr/bin/env python3
"""
Unified experiment runner for all 3 approaches evaluated in the paper "Cost-Aware
Replicas Placement and Tasks Scheduling In Geo-Distributed Infrastructure"
(Si-Mohammed et al.): Heuristic (Section 4.2), Offline COP and Online COP (Section 4.1).

This script does NOT reimplement any scheduling logic. It is a thin subprocess
orchestrator: each method already has its own standalone, self-contained runner
(run_heuristic.py, offline-cop/src/RunOfflineCOP.java, online-cop/run_online_cop.py),
each with its own README explaining how it works. This script just picks the right one,
builds its command line / environment from a single unified CLI, and (optionally)
records the parsed result as one JSON line per run for later aggregation (e.g. building
the paper's comparison figures across methods/instances).

Why subprocesses and not direct imports: the Heuristic and Online COP methods each ship
their OWN copies of job.py / tracker.py / compute_node.py with incompatible internals
(see ../online-cop/README.md) - importing both in the same Python process would collide
in sys.modules (whichever is imported first "wins", silently corrupting the other). Each
subprocess keeps its own module namespace, exactly like running them by hand from the
command line would.

Examples
--------
    # Heuristic (Section 4.2)
    python3 run_experiment.py heuristic ../traces/inst-20J-50N --sigma 0.05

    # Offline COP (Section 4.1, Choco-solver + LNS, whole horizon in one solve)
    python3 run_experiment.py offline ../traces/inst-20J-50N --time-limit 150s

    # Online COP (Section 4.1, Choco-solver + LNS, re-solved per arrival)
    python3 run_experiment.py online ../traces/inst-20J-50N --solver-time-limit 5

    # Append the parsed result of each run to a shared results file for later analysis
    python3 run_experiment.py heuristic ../traces/inst-20J-50N --json-out results.jsonl
    python3 run_experiment.py offline   ../traces/inst-20J-50N --json-out results.jsonl
    python3 run_experiment.py online    ../traces/inst-20J-50N --json-out results.jsonl

`results.jsonl` can then be loaded with `pandas.read_json(path, lines=True)` to compare
methods/instances side by side.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

# This file lives in paper-artifact/exps/ - every method's own folder is a sibling of
# exps/, one level up.
EXPS_DIR = os.path.abspath(os.path.dirname(__file__))
ARTIFACT_ROOT = os.path.dirname(EXPS_DIR)
HEURISTIC_DIR = ARTIFACT_ROOT
OFFLINE_DIR = os.path.join(ARTIFACT_ROOT, "offline-cop")
ONLINE_DIR = os.path.join(ARTIFACT_ROOT, "online-cop")
TRACES_DIR = os.path.join(ARTIFACT_ROOT, "traces")

# Matches the "            key: value" lines every one of the 3 runners prints as its
# final summary block (see run_heuristic.py / RunOfflineCOP.java / run_online_cop.py).
_RESULT_LINE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$")


def resolve_instance(instance_arg: str) -> str:
    """Accept either a bare instance name ("inst-20J-50N", looked up under
    ../traces/), or a path to an instance directory (relative or absolute) - returns
    an ABSOLUTE path so every subprocess (run from its own method's directory, not
    from here) resolves it the same way regardless of cwd."""
    candidate = instance_arg
    if not os.path.isdir(candidate):
        candidate = os.path.join(TRACES_DIR, instance_arg)
    if not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"Instance not found: tried '{instance_arg}' and '{candidate}'. "
            f"Run '{TRACES_DIR}/generate_traces.py' first if traces/ is empty."
        )
    return os.path.abspath(candidate)


def parse_result_block(stdout_text: str) -> dict:
    """Best-effort parse of a runner's final "key: value" summary block into a dict.
    Numeric-looking values are converted to int/float; "true"/"false" to bool; anything
    else stays a string. Non-matching lines (progress/debug output) are ignored."""
    result = {}
    for line in stdout_text.splitlines():
        m = _RESULT_LINE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
            continue
        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass
        result[key] = value
    return result


def run_and_capture(cmd, cwd, env=None):
    """Run a subprocess, streaming its output live (so the underlying tool's own
    progress/debug prints are still visible) while also capturing it to parse the
    final result block afterward."""
    print(f"$ (cwd={cwd}) {' '.join(cmd)}")
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=merged_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {' '.join(cmd)}")
    return "".join(lines)


def ensure_offline_compiled(force: bool = False):
    """Offline COP is plain Java (no build tool) - compile it once into out/, unless
    already compiled or --recompile was passed. Mirrors offline-cop/README.md's manual
    `javac -cp "lib/*" -d out src/*.java` step."""
    out_dir = os.path.join(OFFLINE_DIR, "out")
    already_built = os.path.isdir(out_dir) and any(
        f.endswith(".class") for f in os.listdir(out_dir)
    )
    if already_built and not force:
        return
    os.makedirs(out_dir, exist_ok=True)
    src_files = [f for f in os.listdir(os.path.join(OFFLINE_DIR, "src")) if f.endswith(".java")]
    cmd = ["javac", "-cp", os.path.join(OFFLINE_DIR, "lib", "*"), "-d", "out"] + \
          [os.path.join("src", f) for f in src_files]
    run_and_capture(cmd, cwd=OFFLINE_DIR)


def run_heuristic(args) -> dict:
    """Section 4.2: utility + acceleration factor, threshold `sigma`. No storage
    constraint, no migration/preemption (see ../README.md)."""
    instance = resolve_instance(args.instance)
    cmd = [
        sys.executable, "run_heuristic.py", instance,
        "--sigma", str(args.sigma),
        "--until", str(args.until),
    ]
    stdout_text = run_and_capture(cmd, cwd=HEURISTIC_DIR)
    result = parse_result_block(stdout_text)
    result["method"] = "heuristic"
    result["sigma"] = args.sigma
    return result


def run_offline(args) -> dict:
    """Section 4.1: Choco-solver + Large Neighborhood Search, one CP solve over the
    whole horizon. NP-hard - OFFLINE_COP_TIME_LIMIT bounds the solver's wall-clock
    budget; the best solution found within that budget is reported (see
    `proven_optimal` in the output to know whether it was also proven optimal)."""
    ensure_offline_compiled(force=args.recompile)
    instance = resolve_instance(args.instance)

    cmd = ["java", "-cp", "out:lib/*", "RunOfflineCOP", instance]
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        cmd.append(output_dir)

    env = {"OFFLINE_COP_TIME_LIMIT": args.time_limit}
    stdout_text = run_and_capture(cmd, cwd=OFFLINE_DIR, env=env)
    result = parse_result_block(stdout_text)
    result["method"] = "offline"
    result["time_limit"] = args.time_limit
    return result


def run_online(args) -> dict:
    """Section 4.1: Choco-solver + LNS, re-solved on every job arrival from a
    node-free-time estimate (not the whole horizon) - work already Started is never
    reconsidered. ONLINE_COP_SOLVER_TIME_LIMIT_S bounds EACH solver call's wall-clock
    budget (there can be many calls per instance, see ../online-cop/README.md)."""
    instance = resolve_instance(args.instance)
    cmd = [
        sys.executable, "run_online_cop.py", instance,
        "--until", str(args.until),
        "--solver-time-limit", str(args.solver_time_limit),
    ]
    env = {"ONLINE_COP_OBJECTIVE": str(args.objective)}
    stdout_text = run_and_capture(cmd, cwd=ONLINE_DIR, env=env)
    result = parse_result_block(stdout_text)
    result["method"] = "online"
    result["solver_time_limit"] = args.solver_time_limit
    result["objective"] = args.objective
    return result


def build_parser() -> argparse.ArgumentParser:
    # Shared by every subcommand so `--json-out` can be passed after the method name
    # (e.g. "run_experiment.py heuristic inst-10J-50N --json-out results.jsonl"), which
    # is where argparse expects subcommand-level options to live.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json-out", default=None,
        help="Append the parsed result of this run as one JSON line to this file "
             "(created if missing). Load later with pandas.read_json(path, lines=True).",
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="method", required=True, help="Which approach to run")

    p_heuristic = subparsers.add_parser("heuristic", parents=[common], help="Heuristic approach (Section 4.2)")
    p_heuristic.add_argument("instance", help="Instance name (e.g. inst-20J-50N) or path to an instance dir")
    p_heuristic.add_argument("--sigma", type=float, default=0.05,
                              help="Acceleration threshold (default: 0.05) - smaller accepts replicas more eagerly")
    p_heuristic.add_argument("--until", type=float, default=200000, help="Simulation horizon safety net (s)")
    p_heuristic.set_defaults(func=run_heuristic)

    p_offline = subparsers.add_parser("offline", parents=[common], help="Offline COP approach (Section 4.1)")
    p_offline.add_argument("instance", help="Instance name (e.g. inst-20J-50N) or path to an instance dir")
    p_offline.add_argument("--time-limit", default="150s",
                            help="Solver wall-clock budget, Choco format e.g. '150s'/'5m' (default: 150s)")
    p_offline.add_argument("--output-dir", default=None,
                            help="If set, write works_exec_solution.csv/transfers_solution.csv on every improved solution")
    p_offline.add_argument("--recompile", action="store_true",
                            help="Force javac recompilation even if out/ already has classes")
    p_offline.set_defaults(func=run_offline)

    p_online = subparsers.add_parser("online", parents=[common], help="Online COP approach (Section 4.1)")
    p_online.add_argument("instance", help="Instance name (e.g. inst-20J-50N) or path to an instance dir")
    p_online.add_argument("--solver-time-limit", type=float, default=5.0,
                           help="Solver wall-clock budget PER SOLVE CALL, in seconds (default: 5)")
    p_online.add_argument("--objective", type=int, default=0, choices=[0, 1, 2],
                           help="0=sum flow time (default, comparable to the other methods), "
                                "1=max flow time, 2=new job's own flow time")
    p_online.add_argument("--until", type=float, default=200000, help="Simulation horizon safety net (s)")
    p_online.set_defaults(func=run_online)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    start = time.time()
    result = args.func(args)
    result["wall_clock_s"] = round(time.time() - start, 2)
    result.setdefault("instance", os.path.basename(resolve_instance(args.instance).rstrip("/")))

    print("\n=== parsed result ===")
    print(json.dumps(result, indent=2))

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nAppended to {args.json_out}")


if __name__ == "__main__":
    main()
