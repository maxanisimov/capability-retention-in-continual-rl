"""Parallel multi-seed / multi-method experiment launcher (paper_2503_07671).

FULL PARALLELISM BY DEFAULT. Instead of running the five methods of a seed
sequentially inside one ``run_experiment`` process, this launcher issues one
``run_experiment`` invocation per (env, seed, method-group), all concurrently,
using ``--skip-*`` flags so each invocation trains exactly one method group and
writes to its own method sub-directory. A seed's methods therefore run on
different CPU cores, cutting each seed's wall-clock from sum(methods) to
max(method) (~4-5x on this workload). Set ``SERIAL_METHODS=1`` to restore the
old one-process-per-seed behaviour.

Method groups (each an independent job):
    baselines_lag : PPO-Lagrangian + PPO-PID-Lagrangian  (train_ppo_lagrangian)
    cpo           : CPO
    shielded      : PPO-Shield
    rashomon      : PSPO

Env vars:
    SWEEP_PARALLEL  max concurrent jobs (default: run all at once)
    SEEDS           comma list (default 0..9)
    ENVS            comma list of short env names (default all 6)
    SMOKE_TIMESTEPS override every method's budget (fast dry run)
    SERIAL_METHODS  =1 -> one invocation per seed (methods serial; old behaviour)
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")
OUT_BASE = os.environ.get(
    "PAPER_OUT_BASE",
    "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/no_earlystop",
)
LOGDIR = REPO / OUT_BASE / "_launch_logs"

ENV_PIPELINE = {
    "media_streaming": "paper_2503_07671_media_streaming",
    "colour_bomb": "paper_2503_07671_colour_bomb",
    "colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
    "bridge_crossing": "paper_2503_07671_bridge_crossing",
    "bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
    "mini_pacman": "paper_2503_07671_minipacman",
}

# Each method group runs a single method-stage via skip flags (base PPO always skipped).
# The value is (skip_flags, jobs): baselines_lag uses jobs=2 so PPO-Lagrangian and
# PPO-PID-Lagrangian train as separate processes (own cores) yet still write the
# combined ppo_lagrangian/ output. Other groups are single-method -> jobs=1.
METHOD_JOBS = {
    "baselines_lag": (["--skip-ppo-policy", "--skip-cpo-policy", "--skip-shielded-policy", "--skip-rashomon-policy", "--skip-rashomon-adaptive-policy"], 2),
    "cpo": (["--skip-ppo-policy", "--skip-ppo-lagrangian", "--skip-shielded-policy", "--skip-rashomon-policy", "--skip-rashomon-adaptive-policy"], 1),
    "shielded": (["--skip-ppo-policy", "--skip-ppo-lagrangian", "--skip-cpo-policy", "--skip-rashomon-policy", "--skip-rashomon-adaptive-policy"], 1),
    "rashomon": (["--skip-ppo-policy", "--skip-baselines", "--skip-shielded-policy", "--skip-rashomon-adaptive-policy"], 1),
    # Unsafe PPO reference. Historically no group ran it (every other group
    # passes --skip-ppo-policy), so it is kept OUT of DEFAULT_METHOD_GROUPS
    # below to preserve the default sweep; select it with METHOD_GROUPS=ppo.
    "ppo": (["--skip-ppo-lagrangian", "--skip-cpo-policy", "--skip-shielded-policy", "--skip-rashomon-policy", "--skip-rashomon-adaptive-policy"], 1),
}

# The four groups a bare `run_seed_experiments.py` has always run.
DEFAULT_METHOD_GROUPS = ["baselines_lag", "cpo", "shielded", "rashomon"]

SEEDS = [int(s) for s in os.environ.get("SEEDS", ",".join(map(str, range(10)))).split(",")]
ENVS = os.environ.get("ENVS", ",".join(ENV_PIPELINE)).split(",")
# Restrict to a subset of METHOD_JOBS, e.g. METHOD_GROUPS=shielded to re-run only
# the shielded policy without touching the other methods' output directories.
METHOD_GROUPS = os.environ.get("METHOD_GROUPS", ",".join(DEFAULT_METHOD_GROUPS)).split(",")
SMOKE = os.environ.get("SMOKE_TIMESTEPS")
SERIAL = os.environ.get("SERIAL_METHODS")
NO_PIN = os.environ.get("NO_PIN")

_unknown_groups = sorted(set(METHOD_GROUPS) - set(METHOD_JOBS))
if _unknown_groups:
    raise SystemExit(
        f"Unknown METHOD_GROUPS: {_unknown_groups}. Valid groups: {sorted(METHOD_JOBS)}."
    )


def _cpu_pool() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def build_jobs() -> list[dict]:
    """Build the per-(env, seed, method-group) job list.

    Each job is given its own ``--cpu-ids`` slice. Without this the pipeline
    defaults ``--cpu-ids`` to *every* core and then takes ``cpu_ids[:jobs]``
    (train_policy_optimisation_pipeline.py:389), so every concurrently launched
    job pins itself to CPU 0 and they all contend for one core -- which cost
    roughly an order of magnitude of throughput on past sweeps. Set NO_PIN=1 to
    fall back to that default.
    """

    pool = _cpu_pool()
    jobs: list[dict] = []
    for env in ENVS:
        pipeline = ENV_PIPELINE[env]
        for seed in SEEDS:
            common = [PY, "-m", "projects.safe_policy_optimisation.run_experiment",
                      "--pipeline", pipeline, "--seed", str(seed),
                      "--output-dir", f"{OUT_BASE}/{env}", "--run-id", f"seed{seed}"]
            if SMOKE:
                common = common + ["--total-timesteps", SMOKE]
            if SERIAL:
                # one invocation per seed; parallelise the 3 baselines internally.
                # Note this ignores METHOD_GROUPS: it runs every method in one go.
                jobs.append({"tag": f"{env}/seed{seed}", "cmd": common + ["--skip-ppo-policy", "--jobs", "3"]})
            else:
                for group in METHOD_GROUPS:
                    flags, njobs = METHOD_JOBS[group]
                    jobs.append({"tag": f"{env}/seed{seed}/{group}",
                                 "cmd": common + flags + ["--jobs", str(njobs)],
                                 "n_cpus": int(njobs)})

    if not NO_PIN:
        # CPU_OFFSET lets concurrent launcher invocations use disjoint cores;
        # without it each invocation starts at core 0 and they collide.
        cursor = int(os.environ.get("CPU_OFFSET", "0"))
        for job in jobs:
            n = job.pop("n_cpus", 1)
            ids = [pool[(cursor + k) % len(pool)] for k in range(n)]
            cursor += n
            job["cmd"] = job["cmd"] + ["--cpu-ids", ",".join(map(str, ids))]
        if cursor > len(pool):
            print(f"warning: {cursor} worker slots over {len(pool)} cores; assignments wrap "
                  "and some jobs will share a core.", flush=True)
    return jobs


def run_job(job: dict) -> tuple[str, int, float]:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / (job["tag"].replace("/", "__") + ".log")
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(job["cmd"], cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return job["tag"], rc, time.time() - t0


def main() -> int:
    jobs = build_jobs()
    parallel = int(os.environ.get("SWEEP_PARALLEL", "0")) or len(jobs)
    print(f"launching {len(jobs)} jobs (envs={len(ENVS)} x seeds={len(SEEDS)} x "
          f"{'1 (serial methods)' if SERIAL else str(len(METHOD_GROUPS))+' method-groups'}), "
          f"groups={'all (serial)' if SERIAL else ','.join(METHOD_GROUPS)}, "
          f"parallel={parallel}, smoke={SMOKE or 'off'}", flush=True)
    done = fail = 0
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(run_job, j): j["tag"] for j in jobs}
        for fut in as_completed(futs):
            tag, rc, secs = fut.result()
            done += 1
            fail += int(rc != 0)
            print(f"[{done}/{len(jobs)}] {'ok' if rc == 0 else f'FAIL(rc={rc})'} {tag} ({secs:.0f}s)", flush=True)
    print(f"PAPER_SWEEP_DONE total={len(jobs)} failed={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
