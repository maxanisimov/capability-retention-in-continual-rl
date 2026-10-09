#!/usr/bin/env python3
"""Initialisation ablations for PSPO with line-segment LIDs.

Two variants of the behaviour-cloned initial policy, each otherwise identical to
the line-segment control cohort ``runs/segment_lid`` (region-first, segment
LIDs with 4 sub-segments up to 8, all-safe-logit semantics, margin target 2):

* ``no_entropy``: no safe-action entropy loss (weight 0); the margin loss stays.
* ``no_margin``: no margin loss (weight 0); the entropy loss stays, and fitting
  stops at the feasibility criterion (all-mode margin > 0) instead of margin 2.

Colour Bomb v2 runs as reported in the paper: without random actions. The fixed
environment with ``slip_prob = 0`` has exactly the pre-fix transition matrix,
and the pre-fix shield (the one ``segment_lid`` used) was synthesised for it.

Each (variant, environment) group runs its ten seeds through
``run_pspo_one_env.sh`` on ten pinned CPUs taken from a shared pool; long groups
are scheduled first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch_pspo_multi_env import ONE_ENV_LAUNCHER, REPO, RUNS_ROOT, build_launch_environment  # noqa: E402

from projects.safe_policy_optimisation.utils.pspo_defaults import RASHOMON_N_ITERS  # noqa: E402
from projects.safe_policy_optimisation.utils.pspo_launcher import parse_cpu_ids  # noqa: E402

CONTROL = RUNS_ROOT / "segment_lid" / "two_hidden"
OUTPUT_ROOT = REPO / "artifacts/ablation_studies/pspo_segment_init_ablations"
VARIANTS = {
    "no_entropy": {"bc_safe_action_entropy_weight": 0.0},
    "no_margin": {"bc_margin_loss_weight": 0.0},
}
# Longest first, so the multi-hour groups start immediately.
ENVIRONMENTS = (
    "mini_pacman", "bridge_crossing_v2", "bridge_crossing",
    "colour_bomb_v2", "media_streaming", "colour_bomb",
)
PRE_FIX_CB2_SHIELD = (
    RUNS_ROOT.parent / "inputs/colour_bomb_v2/_pre_slipfix_20261007/shield_q.pt"
)
TASK_OVERRIDES = {
    "colour_bomb_v2": {
        "ENV_KWARGS_OVERRIDE": json.dumps({"slip_prob": 0.0}),
        "SHIELD_PATH_OVERRIDE": str(PRE_FIX_CB2_SHIELD),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, check=True, text=True, capture_output=True).stdout
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(status.strip()), "status": status.splitlines()}


def control_settings(environment: str) -> dict:
    """The control's seed-0 config, which every ablation must match except for the init."""
    config = json.loads((CONTROL / environment / "seed0" / "config.json").read_text())
    if environment in TASK_OVERRIDES and config["shield_sha256"] != sha256(PRE_FIX_CB2_SHIELD):
        raise SystemExit(f"{environment}: the control did not use {PRE_FIX_CB2_SHIELD}")
    return {
        "env_kwargs": config["env_kwargs"], "shield_sha256": config["shield_sha256"],
        "total_timesteps": config["total_timesteps"], "adaptive": config["adaptive"],
    }


class CpuPool:
    """Hands out disjoint CPU blocks, in pool order, as groups finish."""

    def __init__(self, cpus: list[int]):
        self._free = list(cpus)
        self._order = {cpu: index for index, cpu in enumerate(cpus)}
        self._condition = threading.Condition()

    def acquire(self, count: int) -> list[int]:
        with self._condition:
            self._condition.wait_for(lambda: len(self._free) >= count)
            block, self._free = self._free[:count], self._free[count:]
            return block

    def release(self, block: list[int]) -> None:
        with self._condition:
            self._free = sorted(self._free + block, key=self._order.__getitem__)
            self._condition.notify_all()


def group_environment(variant: str, environment: str, seeds: list[int], cpus: list[int], args) -> dict[str, str]:
    env = build_launch_environment(
        environment=environment, seeds=seeds, cpu_ids=cpus, architecture="two_hidden",
        run_name=f"pspo_segment_init_ablations_{variant}", n_iters=RASHOMON_N_ITERS,
        dry_run=args.dry_run, safe_region_shape="segment", **VARIANTS[variant],
    )
    env.update(TASK_OVERRIDES.get(environment, {}))
    env["OUTPUT_BASE"] = str(args.output_root / variant / "two_hidden" / environment)
    if args.smoke:
        env.update(PSPO_SMOKE="1", TOTAL_TIMESTEPS="64", RASHOMON_N_ITERS="1")
    return env


def run_group(variant: str, environment: str, args, pool: CpuPool, log_dir: Path) -> dict:
    cpus = pool.acquire(len(args.seeds))
    started = datetime.now(timezone.utc).isoformat()
    try:
        env = group_environment(variant, environment, args.seeds, cpus, args)
        with (log_dir / f"{variant}_{environment}.log").open("w") as log:
            rc = subprocess.run(["bash", str(ONE_ENV_LAUNCHER)], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    finally:
        pool.release(cpus)
    output = Path(env["OUTPUT_BASE"])
    missing = [seed for seed in args.seeds if not (output / f"seed{seed}" / "metrics.json").exists()]
    result = {"variant": variant, "environment": environment, "cpus": cpus, "rc": rc,
              "started": started, "finished": datetime.now(timezone.utc).isoformat(),
              "missing_seeds": [] if args.dry_run else missing}
    print(json.dumps(result), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--cpu-ids", required=True, help="CPU pool in preference order, e.g. '10-31,40-51'.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Full initial-policy fit, then 64 PPO steps and 1 LID iteration; never use for results.")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    cpus = parse_cpu_ids(args.cpu_ids)
    if len(cpus) < len(args.seeds):
        raise SystemExit(f"the pool needs at least {len(args.seeds)} CPUs")
    if not PRE_FIX_CB2_SHIELD.exists():
        raise SystemExit(f"missing shield: {PRE_FIX_CB2_SHIELD}")

    environments = [environment for environment in ENVIRONMENTS if environment in args.envs]
    groups = [(variant, environment) for environment in environments for variant in args.variants]
    log_dir = args.output_root / "_orchestrator"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "control_cohort": str(CONTROL.parent),
        "control_settings": {environment: control_settings(environment) for environment in environments},
        "variants": {variant: VARIANTS[variant] for variant in args.variants},
        "task_overrides": TASK_OVERRIDES,
        "seeds": args.seeds, "cpu_pool": cpus, "smoke": args.smoke, "dry_run": args.dry_run,
        "groups": [f"{variant}/{environment}" for variant, environment in groups],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (log_dir / f"manifest_{stamp}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    pool = CpuPool(cpus)
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        # Submit in order so that long groups claim CPUs first.
        futures = []
        for group in groups:
            futures.append(executor.submit(run_group, *group, args, pool, log_dir))
            threading.Event().wait(0.5)
        results = [future.result() for future in futures]
    (log_dir / f"results_{stamp}.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = [result for result in results if result["rc"] or result["missing_seeds"]]
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
