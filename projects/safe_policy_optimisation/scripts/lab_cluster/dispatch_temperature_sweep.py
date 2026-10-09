"""Spread the stochastic-deployment temperature sweep across free lab machines.

``run_temperature_sweep.py`` parallelises over the cores of one machine, which
is not enough: the full grid is ~27 core-hours, and the login box it was first
launched on has 5 GB of RAM shared with other users -- four workers there were
OOM-killed. Lab machines have ~14 GB free each, so the memory constraint
disappears once the work is spread out.

The schedulable unit is one **(environment, seed)** pair -- all seven variants
for that pair, which is what ``run_temperature_sweep.py`` evaluates in one
invocation. Units are independent because each writes its own
``<env>/seed<k>/<variant>.json`` under the shared NFS output tree, so no two
hosts ever touch the same file and there is no write contention.

Re-running only dispatches pairs with missing cells, so this doubles as the
resume path after a crash or a partial run.

Typical use:

    export SSH_AUTH_SOCK=/tmp/ma5923-agent.sock
    bash scripts/lab_cluster/probe_hosts.sh --out /tmp/hosts.tsv
    python scripts/lab_cluster/dispatch_temperature_sweep.py --hosts /tmp/hosts.tsv
    python scripts/lab_cluster/dispatch_temperature_sweep.py --hosts /tmp/hosts.tsv --launch
"""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import launch, read_hosts  # noqa: E402
from run_temperature_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    VARIANTS,
)

# Rough per-environment cost of one (env, seed) pair, in units of "steps to
# roll out", used only to schedule the expensive environments first so the
# long pole starts earliest. Numbers are mean greedy/nominal episode length x
# 100 episodes x 8 temperatures x 7 variants.
ENV_WEIGHT = {
    "mini_pacman": 56,
    "bridge_crossing_v2": 33,
    "bridge_crossing": 30,
    "colour_bomb_v2": 14,
    "colour_bomb": 4,
    "media_streaming": 2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hosts", type=Path, required=True, help="probe_hosts.sh TSV")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dispatch-dir",
        type=Path,
        default=REPO
        / "projects/safe_policy_optimisation/artifacts/paper_2503_07671"
        / "analysis/temperature_sweep_dispatch",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--workers-per-host",
        type=int,
        default=4,
        help=(
            "Worker processes per host (default 4). Each holds a torch process "
            "plus one checkpoint; lab machines have ~14 GB free, so 4 is "
            "comfortable and leaves the box usable for others."
        ),
    )
    parser.add_argument(
        "--max-hosts", type=int, default=64,
        help="Cap on machines used, to stay a good citizen (default 64).",
    )
    parser.add_argument("--min-free-cores", type=int, default=4)
    parser.add_argument("--mem-per-job-gb", type=float, default=1.5)
    parser.add_argument("--ssh-timeout", type=int, default=30)
    parser.add_argument(
        "--launch", action="store_true",
        help="Actually dispatch. Without it, print the plan and exit.",
    )
    return parser.parse_args(argv)


def pending_pairs(sweep_dir: Path, seeds: int) -> list[tuple[str, int]]:
    """(env, seed) pairs with at least one variant cell still missing."""
    import matplotlib

    matplotlib.use("Agg")
    from plot_extended_budget_learning_curves import ENVIRONMENTS

    pairs: list[tuple[str, int]] = []
    for environment in ENVIRONMENTS:
        for seed in range(seeds):
            missing = [
                spec.key
                for spec in VARIANTS
                if not (
                    sweep_dir / environment.key / f"seed{seed}" / f"{spec.key}.json"
                ).is_file()
            ]
            if missing:
                pairs.append((environment.key, seed))
    # Most expensive environments first: they set the finish time.
    pairs.sort(key=lambda pair: -ENV_WEIGHT.get(pair[0], 1))
    return pairs


def build_script(
    pairs: list[tuple[str, int]], *, workers: int, sweep_dir: Path, log: Path
) -> str:
    """One bash script running this host's share of the pairs, in sequence."""
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {shlex.quote(str(REPO))} || exit 1",
        "echo \"host $(hostname) starting $(date -Is)\"",
    ]
    for env_key, seed in pairs:
        command = (
            f".venv/bin/python "
            f"projects/safe_policy_optimisation/scripts/run_temperature_sweep.py "
            f"--environment {env_key} --seed {seed} "
            f"--reset-seed-mode uniform --workers {workers} "
            f"--output-dir {shlex.quote(str(sweep_dir))}"
        )
        lines += [
            f"echo \"=== {env_key}/seed{seed} $(date -Is) ===\"",
            command,
        ]
    lines.append("echo \"host $(hostname) finished $(date -Is)\"")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = pending_pairs(args.sweep_dir, args.seeds)
    if not pairs:
        print("nothing pending - every (environment, seed) pair is complete")
        return 0

    hosts = read_hosts(args.hosts, args.min_free_cores, args.mem_per_job_gb)
    if not hosts:
        raise SystemExit("no usable hosts in the probe file")
    hosts = hosts[: args.max_hosts]

    # Round-robin the cost-sorted pairs so each host gets a mix of expensive
    # and cheap environments rather than one host drawing every MiniPacman.
    assignment: dict[str, list[tuple[str, int]]] = {host.name: [] for host in hosts}
    for index, pair in enumerate(pairs):
        assignment[hosts[index % len(hosts)].name].append(pair)
    assignment = {name: items for name, items in assignment.items() if items}

    print(f"{len(pairs)} pending (environment, seed) pair(s)")
    print(
        f"{len(assignment)} host(s) x {args.workers_per_host} worker(s) = "
        f"{len(assignment) * args.workers_per_host} cores"
    )
    for name, items in list(assignment.items())[:8]:
        preview = ", ".join(f"{env}/s{seed}" for env, seed in items[:4])
        more = "" if len(items) <= 4 else f" (+{len(items) - 4})"
        print(f"  {name:12s} {len(items):2d} pair(s): {preview}{more}")
    if len(assignment) > 8:
        print(f"  ... and {len(assignment) - 8} more host(s)")

    if not args.launch:
        print("\nplan only; re-run with --launch to dispatch")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dispatch_dir = args.dispatch_dir / stamp
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    started = 0
    for host in hosts:
        items = assignment.get(host.name)
        if not items:
            continue
        script_path = dispatch_dir / f"{host.name}.sh"
        script_path.write_text(
            build_script(
                items,
                workers=args.workers_per_host,
                sweep_dir=args.sweep_dir,
                log=dispatch_dir / f"{host.name}.log",
            ),
            encoding="utf-8",
        )
        if launch(host, script_path, dispatch_dir, args.ssh_timeout):
            started += 1
            print(f"  started {host.name} ({len(items)} pair(s))")
    print(f"\ndispatched {started}/{len(assignment)} host(s)")
    print(f"logs: {dispatch_dir}")
    return 0 if started else 1


if __name__ == "__main__":
    raise SystemExit(main())
