"""BC-margin sweep for the non-media paper MASA environments.

This launcher mirrors the media-streaming margin sweep, but for the remaining
paper MASA environments with available shield inputs.  It builds BC/Rashomon
sets for the requested margins, then runs:

* precomputed PSPO from each margin-specific Rashomon set;
* adaptive PSPO with ``unsafe_update_strategy=rashomon_project``.

It intentionally does not run the line-search adaptive method.

The script is resumable: jobs whose expected terminal artifact already exists
are skipped unless ``FORCE=1`` is set.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")

SET_STAGE = str(REPO / "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py")
PRECOMPUTED_STAGE = str(REPO / "projects/safe_policy_optimisation/stages/train_pspo_precomputed.py")
ADAPTIVE_STAGE = str(REPO / "projects/safe_policy_optimisation/stages/train_pspo_adaptive.py")

OUT_BASE = Path(
    os.environ.get(
        "MASA_MARGIN_OUT_BASE",
        "projects/safe_policy_optimisation/artifacts/bc_margin_sweep/masa_other",
    )
)
LOGDIR = REPO / OUT_BASE / "_launch_logs"

ENV_PIPELINE = {
    "colour_bomb": "paper_2503_07671_colour_bomb",
    "colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
    "bridge_crossing": "paper_2503_07671_bridge_crossing",
    "bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
    "mini_pacman": "paper_2503_07671_minipacman",
}

MARGINS = [float(x) for x in os.environ.get("MARGINS", "0.1,1,2,5,10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", ",".join(map(str, range(10)))).split(",")]
ENVS = [x for x in os.environ.get("ENVS", ",".join(ENV_PIPELINE)).split(",") if x]
PHASES = [x for x in os.environ.get("PHASES", "sets,precomputed,adaptive_project").split(",") if x]

FORCE = os.environ.get("FORCE") == "1"
SMOKE = os.environ.get("SMOKE_TIMESTEPS")
ADAPTIVE_N_ITERS = os.environ.get("ADAPTIVE_N_ITERS", "100")
ADAPTIVE_GRANULARITY = os.environ.get("ADAPTIVE_GRANULARITY", "gradient_step")
DEVICE = os.environ.get("DEVICE", "cpu")


def margin_tag(margin: float) -> str:
    return ("margin_" + ("%g" % margin).replace(".", "p").replace("-", "m"))


def _resolve_env_settings(pipeline_name: str) -> dict:
    from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings

    settings, _pipeline, _task = compose_pipeline_settings(pipeline_name)
    return settings


def common_training_args(cfg: dict, seed: int, total_timesteps: int | str) -> list[str]:
    return [
        "--env-id",
        cfg["env_id"],
        "--env-kwargs",
        json.dumps(cfg["env_kwargs"] or {}),
        "--max-episode-steps",
        str(cfg["max_episode_steps"]),
        "--cost-limit",
        str(cfg["cost_limit"]),
        "--total-timesteps",
        str(total_timesteps),
        "--eval-episodes",
        str(cfg["eval_episodes"]),
        "--seed",
        str(seed),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--n-steps",
        str(cfg["n_steps"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--n-epochs",
        str(cfg["n_epochs"]),
        "--gamma",
        str(cfg["gamma"]),
        "--gae-lambda",
        str(cfg["gae_lambda"]),
        "--clip-range",
        str(cfg["clip_range"]),
        "--ent-coef",
        str(cfg["ent_coef"]),
        "--vf-coef",
        str(cfg["vf_coef"]),
        "--max-grad-norm",
        str(cfg["max_grad_norm"]),
        "--device",
        DEVICE,
        "--early-stop-eval-policy",
        "unshielded",
        "--evaluation-policy",
        str(cfg.get("rashomon_evaluation_policy", "unshielded")),
        "--early-stop-eval-freq",
        str(cfg["early_stop_eval_freq"]),
        "--early-stop-eval-episodes",
        str(cfg["early_stop_eval_episodes"]),
        "--early-stop-success-rate",
        str(cfg["early_stop_success_rate"]),
        "--success-reward-threshold",
        str(cfg["success_reward_threshold"]),
        "--curve-eval-freq",
        str(cfg["curve_eval_freq"]),
        "--curve-eval-episodes",
        str(cfg["curve_eval_episodes"]),
    ]


def build_jobs() -> list[dict]:
    unknown_envs = sorted(set(ENVS) - set(ENV_PIPELINE))
    if unknown_envs:
        raise SystemExit(f"Unknown ENVS={unknown_envs}; valid={sorted(ENV_PIPELINE)}")
    unknown_phases = sorted(set(PHASES) - {"sets", "precomputed", "adaptive_project"})
    if unknown_phases:
        raise SystemExit(
            f"Unknown PHASES={unknown_phases}; valid=['sets','precomputed','adaptive_project']"
        )

    jobs: list[dict] = []
    for env in ENVS:
        cfg = _resolve_env_settings(ENV_PIPELINE[env])
        total_timesteps = SMOKE or cfg["total_timesteps"]
        env_base = OUT_BASE / env
        for margin in MARGINS:
            tag = margin_tag(margin)
            set_dir = env_base / "sets_full_ckpt1" / tag
            if "sets" in PHASES:
                terminal = set_dir / "summary.json"
                cmd = [
                    PY,
                    SET_STAGE,
                    "--shield-path",
                    str(cfg["shield_path"]),
                    "--env-id",
                    cfg["env_id"],
                    "--env-kwargs",
                    json.dumps(cfg["env_kwargs"] or {}),
                    "--state-representation",
                    "features",
                    "--output-dir",
                    str(env_base / "sets_full_ckpt1"),
                    "--run-id",
                    tag,
                    "--seed",
                    "0",
                    "--device",
                    DEVICE,
                    "--hidden-dim",
                    str(cfg["hidden_dim"]),
                    "--n-hidden",
                    str(cfg["n_hidden"]),
                    "--bc-target-margin",
                    str(margin),
                    "--rashomon-n-iters",
                    str(cfg["rashomon_n_iters"]),
                    "--rashomon-checkpoint",
                    "1",
                    "--rashomon-batch-size",
                    str(cfg["rashomon_batch_size"]),
                    "--certificate-samples",
                    str(cfg["certificate_samples"]),
                ]
                jobs.append(
                    {
                        "phase": "sets",
                        "tag": f"{env}/{tag}/sets",
                        "cmd": cmd,
                        "terminal": terminal,
                    }
                )
            if "precomputed" in PHASES:
                for seed in SEEDS:
                    run_root = env_base / "runs" / "precomputed_full" / tag
                    terminal = run_root / f"seed{seed}" / "metrics.json"
                    cmd = [
                        PY,
                        PRECOMPUTED_STAGE,
                        "--rashomon-dir",
                        str(set_dir),
                        "--shield-path",
                        str(cfg["shield_path"]),
                        *common_training_args(cfg, seed, total_timesteps),
                        "--output-dir",
                        str(run_root),
                        "--run-id",
                        f"seed{seed}",
                    ]
                    jobs.append(
                        {
                            "phase": "precomputed",
                            "tag": f"{env}/{tag}/precomputed/seed{seed}",
                            "cmd": cmd,
                            "terminal": terminal,
                            "requires": set_dir / "summary.json",
                        }
                    )
            if "adaptive_project" in PHASES:
                for seed in SEEDS:
                    run_root = env_base / "runs" / "adaptive_project" / tag
                    terminal = run_root / f"seed{seed}" / "metrics.json"
                    cmd = [
                        PY,
                        ADAPTIVE_STAGE,
                        "--base-policy-path",
                        str(set_dir / "base_policy.pt"),
                        "--shield-path",
                        str(cfg["shield_path"]),
                        *common_training_args(cfg, seed, total_timesteps),
                        "--adaptive-granularity",
                        ADAPTIVE_GRANULARITY,
                        "--unsafe-update-strategy",
                        "rashomon_project",
                        "--rashomon-n-iters",
                        ADAPTIVE_N_ITERS,
                        "--output-dir",
                        str(run_root),
                        "--run-id",
                        f"seed{seed}",
                    ]
                    jobs.append(
                        {
                            "phase": "adaptive_project",
                            "tag": f"{env}/{tag}/adaptive_project/seed{seed}",
                            "cmd": cmd,
                            "terminal": terminal,
                            "requires": set_dir / "base_policy.pt",
                        }
                    )
    return jobs


def run_job(job: dict) -> tuple[str, str, int, float, str]:
    terminal = REPO / job["terminal"]
    if terminal.exists() and not FORCE:
        return job["phase"], job["tag"], 0, 0.0, "skip"
    required = job.get("requires")
    if required is not None and not (REPO / required).exists():
        return job["phase"], job["tag"], 86, 0.0, f"missing prerequisite {required}"

    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
    )
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / (job["tag"].replace("/", "__") + ".log")
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(job["cmd"], cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return job["phase"], job["tag"], rc, time.time() - t0, str(log)


def run_phase(jobs: list[dict], phase: str, parallel: int) -> int:
    phase_jobs = [j for j in jobs if j["phase"] == phase]
    if not phase_jobs:
        return 0
    print(f"launching phase={phase} jobs={len(phase_jobs)} parallel={parallel}", flush=True)
    done = fail = 0
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(run_job, j): j["tag"] for j in phase_jobs}
        for fut in as_completed(futs):
            _phase, tag, rc, secs, detail = fut.result()
            done += 1
            fail += int(rc != 0)
            status = "skip" if detail == "skip" else ("ok" if rc == 0 else f"FAIL(rc={rc})")
            print(f"[{done}/{len(phase_jobs)}] {status} {tag} {secs:.0f}s {detail}", flush=True)
    print(f"phase={phase} done={done} failed={fail}", flush=True)
    return fail


def main() -> int:
    jobs = build_jobs()
    set_parallel = int(os.environ.get("SET_PARALLEL", "5"))
    pspo_parallel = int(os.environ.get("PSPO_PARALLEL", "20"))
    fail = 0
    for phase in ("sets", "precomputed", "adaptive_project"):
        if phase not in PHASES:
            continue
        parallel = set_parallel if phase == "sets" else pspo_parallel
        fail += run_phase(jobs, phase, parallel)
        if fail:
            break
    print(f"MASA_MARGIN_SWEEP_DONE failed={fail}", flush=True)
    return int(fail != 0)


if __name__ == "__main__":
    raise SystemExit(main())
