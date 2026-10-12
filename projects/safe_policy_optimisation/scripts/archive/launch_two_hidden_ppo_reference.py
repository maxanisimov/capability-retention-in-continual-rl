"""Launch missing two-hidden single-method reference runs with isolated CPU groups.

This launcher is intentionally separate from ``run_seed_experiments.py``.  It
solves the orchestration problem:

* choose currently low-load CPU cores from this process' affinity mask;
* split them into one non-overlapping CPU group per environment;
* start one detached ``screen`` session per environment;
* inside each screen, run ``run_seed_experiments.py`` with the requested
  ``METHOD_GROUPS`` value.

The worker launcher still handles one job per seed and pins those seed jobs
within the CPU affinity group inherited through ``taskset``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PYTHON = REPO / ".venv/bin/python"


ENV_SPECS = {
    "bridge_crossing": {
        "screen_suffix": "bc-v1",
        "out_base": "outputs/_sweeps_2hidden_bridge_crossing_baselines_only",
        "label": "Bridge Crossing v1",
    },
    "bridge_crossing_v2": {
        "screen_suffix": "bc-v2",
        "out_base": "outputs/_sweeps_2hidden_bridge_crossing_v2_baselines_only",
        "label": "Bridge Crossing v2",
    },
    "colour_bomb": {
        "screen_suffix": "colour-v1",
        "out_base": "outputs/_sweeps_2hidden_colour_bomb_baselines_only",
        "label": "Colour Bomb v1",
    },
    "media_streaming": {
        "screen_suffix": "media",
        "out_base": "outputs/_sweeps_2hidden_media_streaming_baselines_only",
        "label": "Media Streaming",
    },
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    value = value.strip()
    if re.fullmatch(r"\d+-\d+", value):
        lo, hi = [int(x) for x in value.split("-", 1)]
        if hi < lo:
            raise argparse.ArgumentTypeError(f"invalid descending seed range: {value}")
        return list(range(lo, hi + 1))
    try:
        return [int(x) for x in parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed list: {value}") from exc


def current_cpu_load() -> dict[int, float]:
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
    else:
        allowed = list(range(os.cpu_count() or 1))
    load = {cpu: 0.0 for cpu in allowed}
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "psr,pcpu", "--no-headers"],
            text=True,
        )
    except Exception:
        return load
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            cpu = int(parts[0])
            pcpu = float(parts[1])
        except ValueError:
            continue
        if cpu in load:
            load[cpu] += pcpu
    return load


def choose_cpu_groups(n_groups: int, cpus_per_group: int) -> list[list[int]]:
    load = current_cpu_load()
    needed = n_groups * cpus_per_group
    if len(load) < needed:
        raise SystemExit(
            f"Need {needed} CPUs in the current affinity mask, found {len(load)}."
        )
    chosen = [cpu for cpu, _ in sorted(load.items(), key=lambda item: (item[1], item[0]))[:needed]]
    return [chosen[i : i + cpus_per_group] for i in range(0, needed, cpus_per_group)]


def parse_cpu_groups(value: str, expected: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for group in value.split(";"):
        cpus = [int(x.strip()) for x in group.split(",") if x.strip()]
        if cpus:
            groups.append(cpus)
    if len(groups) != expected:
        raise argparse.ArgumentTypeError(
            f"expected {expected} CPU groups separated by ';', got {len(groups)}"
        )
    flattened = [cpu for group in groups for cpu in group]
    if len(flattened) != len(set(flattened)):
        raise argparse.ArgumentTypeError("CPU groups overlap")
    return groups


def existing_screen_names() -> set[str]:
    try:
        output = subprocess.check_output(["screen", "-ls"], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output
    names: set[str] = set()
    for line in output.splitlines():
        match = re.search(r"\d+\.([^\s\t]+)", line)
        if match:
            names.add(match.group(1))
    return names


def launch_env(
    *,
    env_name: str,
    method_group: str,
    seeds: list[int],
    cpu_group: list[int],
    screen_prefix: str,
    log_dir: Path,
    python: Path,
    sweep_parallel: int,
    dry_run: bool,
) -> None:
    spec = ENV_SPECS[env_name]
    screen_name = f"{screen_prefix}-{spec['screen_suffix']}"
    logfile = log_dir / f"{env_name}.log"
    cpu_csv = ",".join(str(cpu) for cpu in cpu_group)
    seed_csv = ",".join(str(seed) for seed in seeds)

    env = os.environ.copy()
    env.update(
        {
            "METHOD_GROUPS": method_group,
            "SEEDS": seed_csv,
            "SWEEP_PARALLEL": str(sweep_parallel),
            "PAPER_OUT_BASE": spec["out_base"],
            "ENVS": env_name,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    cmd = [
        "screen",
        "-dmS",
        screen_name,
        "-L",
        "-Logfile",
        str(logfile),
        "taskset",
        "-c",
        cpu_csv,
        str(python),
        "projects/safe_policy_optimisation/scripts/run_seed_experiments.py",
    ]
    print(
        f"{spec['label']}: method_group={method_group} screen={screen_name} "
        f"cpus={cpu_csv} seeds={seed_csv} "
        f"out={spec['out_base']}"
    )
    if dry_run:
        print("  DRY RUN:", " ".join(cmd))
        return
    subprocess.check_call(cmd, cwd=REPO, env=env)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        default="bridge_crossing,bridge_crossing_v2,colour_bomb,media_streaming",
        help=f"Comma-separated env names. Valid: {','.join(sorted(ENV_SPECS))}",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0-9"))
    parser.add_argument(
        "--method-group",
        choices=["ppo", "shielded"],
        default="ppo",
        help=(
            "Method group to run through run_seed_experiments.py. "
            "'shielded' trains PPO-Shield and evaluates both shielded and "
            "nominal/unshielded policies."
        ),
    )
    parser.add_argument("--cpus-per-env", type=int, default=10)
    parser.add_argument(
        "--cpu-groups",
        default=None,
        help="Optional explicit CPU groups, e.g. '0,1,2;10,11,12'. "
        "If omitted, choose least-loaded CPUs from current affinity.",
    )
    parser.add_argument(
        "--sweep-parallel",
        type=int,
        default=None,
        help="Parallel seed jobs per environment. Default: min(len(seeds), cpus-per-env).",
    )
    parser.add_argument(
        "--screen-prefix",
        default=None,
        help="Prefix for per-environment screen session names.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-existing-screens",
        action="store_true",
        help="Do not fail if a target screen name already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    envs = parse_csv(args.envs)
    unknown = sorted(set(envs) - set(ENV_SPECS))
    if unknown:
        raise SystemExit(f"Unknown env(s): {unknown}. Valid: {sorted(ENV_SPECS)}")
    if not envs:
        raise SystemExit("No environments requested.")
    if args.cpus_per_env <= 0:
        raise SystemExit("--cpus-per-env must be positive.")
    if args.sweep_parallel is not None and args.sweep_parallel <= 0:
        raise SystemExit("--sweep-parallel must be positive.")
    if not args.python.exists():
        raise SystemExit(f"Python executable not found: {args.python}")
    screen_prefix = args.screen_prefix or f"pspo-{args.method_group}-2h"
    log_dir = args.log_dir or REPO / f"outputs/_sweeps_2hidden_{args.method_group}_launch_logs"

    if args.cpu_groups:
        cpu_groups = parse_cpu_groups(args.cpu_groups, expected=len(envs))
    else:
        cpu_groups = choose_cpu_groups(len(envs), args.cpus_per_env)
    sweep_parallel = args.sweep_parallel or min(len(args.seeds), args.cpus_per_env)

    if not args.allow_existing_screens and not args.dry_run:
        existing = existing_screen_names()
        requested = {
            f"{screen_prefix}-{ENV_SPECS[env]['screen_suffix']}" for env in envs
        }
        collisions = sorted(existing & requested)
        if collisions:
            raise SystemExit(
                "Refusing to launch because these screen sessions already exist: "
                + ", ".join(collisions)
                + ". Use --allow-existing-screens if this is intentional."
            )

    log_dir.mkdir(parents=True, exist_ok=True)
    for env_name, cpu_group in zip(envs, cpu_groups, strict=True):
        launch_env(
            env_name=env_name,
            method_group=args.method_group,
            seeds=args.seeds,
            cpu_group=cpu_group,
            screen_prefix=screen_prefix,
            log_dir=log_dir,
            python=args.python,
            sweep_parallel=sweep_parallel,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        subprocess.run(["screen", "-ls"], cwd=REPO, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
