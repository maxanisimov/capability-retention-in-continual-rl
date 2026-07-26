"""PSPO Rashomon-set-size sweep launcher (single ``rashomon_n_iters`` per run).

Trains ONLY the ``rashomon_policy`` (PSPO) stage for every (env, seed) at one
Rashomon-synthesis iteration count, building a fresh Rashomon set per env at that
count. Companion to ``run_seed_experiments.py``; used to extend the 2k/10k
Rashomon-set-size ablation to 20k / 50k / 100k.

Why a dedicated launcher (vs. run_seed_experiments.py):
  * It overrides ``--rashomon-n-iters`` AND ``--rashomon-dir`` per run. The
    Rashomon set is a cached precompute (train_policy_optimisation_pipeline.py
    reuses ``<rashomon-dir>/rashomon_param_bounds.pt`` if present), so a fresh
    per-iter-count dir is required or the existing 2k set is silently reused.
  * It gates seed0 (the warmup that builds the set) ahead of seeds 1-9 PER ENV,
    so the 10 seeds of an env never race to build the same dir, while fast envs
    still fan out without waiting for slow ones (mini_pacman set build dominates).

The base pipelines ``paper_2503_07671_<env>`` are reused unchanged; each job adds
the rashomon-only skip flags and the two overrides. No settings YAML is edited.

Env vars:
    RASHOMON_N_ITERS  required, e.g. 20000
    ITER_TAG          required, e.g. rashomon20k -> artifact path stem
    SEEDS             comma list (default 0..9); SEEDS[0] is the per-env warmup
    ENVS              comma list of short env names (default all 6)
    SWEEP_PARALLEL    max concurrent jobs (default 20)
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")

RASHOMON_N_ITERS = os.environ.get("RASHOMON_N_ITERS")
ITER_TAG = os.environ.get("ITER_TAG")
if not RASHOMON_N_ITERS or not ITER_TAG:
    raise SystemExit("RASHOMON_N_ITERS and ITER_TAG must both be set (e.g. 20000 / rashomon20k).")

ART = f"projects/safe_policy_optimisation/artifacts/paper_2503_07671_{ITER_TAG}"
OUT_BASE = f"{ART}_experiments"          # per-seed PSPO training outputs
SET_BASE = ART                            # fresh Rashomon sets: {SET_BASE}/<env>/rashomon_fullcov
LOGDIR = REPO / OUT_BASE / "_launch_logs"

# Short env name -> base (2k default) pipeline key. Reused verbatim from
# run_seed_experiments.py; these pipelines include colour_bomb_v2 and default all
# stages, so we skip everything except rashomon_policy below.
ENV_PIPELINE = {
    "media_streaming": "paper_2503_07671_media_streaming",
    "colour_bomb": "paper_2503_07671_colour_bomb",
    "colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
    "bridge_crossing": "paper_2503_07671_bridge_crossing",
    "bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
    "mini_pacman": "paper_2503_07671_minipacman",
}

RASHOMON_ONLY = ["--skip-ppo-policy", "--skip-baselines", "--skip-shielded-policy"]

SEEDS = [int(s) for s in os.environ.get("SEEDS", ",".join(map(str, range(10)))).split(",")]
ENVS = os.environ.get("ENVS", ",".join(ENV_PIPELINE)).split(",")


def make_job(env: str, seed: int) -> dict:
    pipeline = ENV_PIPELINE[env]
    cmd = [
        PY, "-m", "projects.safe_policy_optimisation.run_experiment",
        "--pipeline", pipeline, "--seed", str(seed),
        "--output-dir", f"{OUT_BASE}/{env}", "--run-id", f"seed{seed}",
        "--rashomon-n-iters", str(RASHOMON_N_ITERS),
        "--rashomon-dir", f"{SET_BASE}/{env}/rashomon_fullcov",
        *RASHOMON_ONLY, "--jobs", "1",
    ]
    return {"tag": f"{env}/seed{seed}", "cmd": cmd, "env": env, "seed": seed}


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
    parallel = int(os.environ.get("SWEEP_PARALLEL", "20")) or 20
    warmup_seed, rest_seeds = SEEDS[0], SEEDS[1:]
    total = len(ENVS) * len(SEEDS)
    print(f"launching {total} PSPO jobs (n_iters={RASHOMON_N_ITERS}, tag={ITER_TAG}): "
          f"envs={len(ENVS)} x seeds={len(SEEDS)}, warmup_seed={warmup_seed}, "
          f"parallel={parallel}", flush=True)

    done = fail = 0
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        meta = {}  # future -> job
        for env in ENVS:  # phase 1: submit every env's warmup (builds its Rashomon set)
            f = ex.submit(run_job, make_job(env, warmup_seed))
            meta[f] = {"env": env, "seed": warmup_seed}
        pending = set(meta)

        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                tag, rc, secs = fut.result()
                info = meta.pop(fut)
                done += 1
                fail += int(rc != 0)
                print(f"[{done}/{total}] {'ok' if rc == 0 else f'FAIL(rc={rc})'} {tag} ({secs:.0f}s)", flush=True)
                # phase 2: a completed warmup unlocks its env's remaining seeds.
                if info["seed"] == warmup_seed:
                    if rc != 0:
                        print(f"  [skip fan-out] {info['env']}: warmup seed{warmup_seed} failed "
                              f"(rc={rc}); its Rashomon set may be absent — not launching seeds "
                              f"{rest_seeds} to avoid a build race.", flush=True)
                        continue
                    for seed in rest_seeds:
                        f = ex.submit(run_job, make_job(info["env"], seed))
                        meta[f] = {"env": info["env"], "seed": seed}
                        pending.add(f)

    print(f"RASHOMON_ITER_SWEEP_DONE tag={ITER_TAG} n_iters={RASHOMON_N_ITERS} "
          f"total={total} failed={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
