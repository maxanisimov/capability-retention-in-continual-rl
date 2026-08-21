"""AdaptiveSafePPO multi-seed launcher (paper_2503_07671 MASA environments).

Runs the unified ``train_pspo_adaptive`` method
for every (env, seed) pair, fully in parallel, with **each job pinned to its
own CPU core** via ``taskset`` (plus single-threaded BLAS) to maximise
parallelism without cross-job contention.

Companion to ``run_seed_experiments.py`` (which sweeps the baseline methods via
``run_experiment``); this launcher invokes the adaptive stage directly because
the method is not part of the pipeline orchestrator. Per-env settings
(env kwargs, shield path, PPO hyperparameters, budgets, eval settings) are
resolved from the same ``paper_2503_07671`` YAML the other methods use, so the
runs are directly comparable. The safe base policy is the BC-fit
``base_policy.pt`` from each env's precomputed Rashomon directory (the base
policy is independent of the offline set's iteration count).

Outputs (kept separate from the other methods' artifact trees):
    artifacts/paper_2503_07671/runs/adaptive/<env>/seed<k>/
        metrics.json  summary.json  config.json  model.zip  episodes.csv ...

Env vars:
    SEEDS             comma list (default 0..9)
    ENVS              comma list of short env names (default all 6)
    ADAPTIVE_N_ITERS  per-computation Rashomon budget (default 100)
    ADAPTIVE_VERIFY_FIRST  false (default) for region-first; true for
                       verify-then-project behavior.
    ADAPTIVE_FREQ     update (default), rollout, once, or every-N-rollouts.
    ADAPTIVE_DIRECTIONAL  true (default) or false.
    ADAPTIVE_SURROGATE  logsumexp (default) or probability.
    ADAPTIVE_REGION_MODE  replace (default) or union.
    ADAPTIVE_VERSION  deprecated compatibility alias: v1/v2.
    ADAPTIVE_GRANULARITY  deprecated compatibility alias.
    ADAPTIVE_STRATEGY  rashomon_project (default) or none.
                       'none' is the BC-init-without-projection ablation: the
                       policy is BC-initialised and explores under the shield,
                       every update is verified, but nothing is ever projected
                       or reverted. Reports safe_update_fraction; provides no
                       safety guarantee. See settings/ablation_studies/.
    SMOKE_TIMESTEPS   override every env's budget (fast dry run)
    NO_PIN            =1 -> disable taskset core pinning
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
STAGE = str(REPO / "projects/safe_policy_optimisation/stages/train_pspo_adaptive.py")

OUT_BASE = os.environ.get(
    "ADAPTIVE_OUT_BASE",
    "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/pspo_adaptive",
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

SEEDS = [int(s) for s in os.environ.get("SEEDS", ",".join(map(str, range(10)))).split(",")]
ENVS = os.environ.get("ENVS", ",".join(ENV_PIPELINE)).split(",")
ADAPTIVE_N_ITERS = os.environ.get("ADAPTIVE_N_ITERS", "100")
ADAPTIVE_TOTAL_ITERS = os.environ.get("ADAPTIVE_TOTAL_ITERS")
ADAPTIVE_VERSION = os.environ.get("ADAPTIVE_VERSION", "v2")
ADAPTIVE_REGION_MODE = os.environ.get("ADAPTIVE_REGION_MODE", "replace")
ADAPTIVE_GRANULARITY = os.environ.get("ADAPTIVE_GRANULARITY", "gradient_step")
ADAPTIVE_STRATEGY = os.environ.get("ADAPTIVE_STRATEGY", "rashomon_project")
ADAPTIVE_VERIFY_FIRST = os.environ.get(
    "ADAPTIVE_VERIFY_FIRST", "true" if ADAPTIVE_VERSION == "v1" else "false"
)
ADAPTIVE_FREQ = os.environ.get(
    "ADAPTIVE_FREQ", "rollout" if ADAPTIVE_GRANULARITY == "train_phase" else "update"
)
ADAPTIVE_DIRECTIONAL = os.environ.get("ADAPTIVE_DIRECTIONAL", "true")
ADAPTIVE_SURROGATE = os.environ.get("ADAPTIVE_SURROGATE", "logsumexp")
SMOKE = os.environ.get("SMOKE_TIMESTEPS")
NO_PIN = os.environ.get("NO_PIN")


def _resolve_env_settings(pipeline_name: str) -> dict:
    """Flat argparse-style settings for a paper pipeline (same YAML as PSPO runs)."""
    from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings

    settings, _pipeline, _task = compose_pipeline_settings(pipeline_name)
    return settings


def _canonical_base_dir(environment: str) -> Path:
    output_root = Path(OUT_BASE)
    if not output_root.is_absolute():
        output_root = REPO / output_root
    return output_root / "_base_policies" / environment


def _prepare_all_safe_base_policy(environment: str, cfg: dict) -> Path:
    """Create/reuse the all-safe-logit base required by canonical PSPO adaptive."""

    from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
        load_shield_mask,
        make_safe_behaviour_payload,
    )
    from projects.safe_policy_optimisation.utils.pspo_adaptive_launcher import (
        base_policy_artifact_matches,
    )

    base_dir = _canonical_base_dir(environment)
    shield_path = Path(cfg["shield_path"])
    if not shield_path.is_absolute():
        shield_path = REPO / shield_path
    mask = load_shield_mask(shield_path)
    _, metadata = make_safe_behaviour_payload(mask)
    dataset_size = int(metadata["dataset_size"])
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_hidden = int(cfg.get("n_hidden", 2))
    target_margin = float(cfg.get("bc_target_margin", 2.0))
    if not base_policy_artifact_matches(
        base_dir,
        shield_path=shield_path,
        dataset_size=dataset_size,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        state_representation="one_hot_discrete_observation",
        margin_mode="all",
        target_margin=target_margin,
    ):
        command = [
            PY,
            str(
                REPO
                / "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py"
            ),
            "--output-dir", str(base_dir.parent),
            "--run-id", base_dir.name,
            "--base-policy-only",
            "--shield-path", str(shield_path),
            "--env-id", str(cfg["env_id"]),
            "--env-kwargs", json.dumps(cfg.get("env_kwargs") or {}),
            "--state-representation", "one_hot",
            "--seed", "0",
            "--device", str(cfg.get("device", "cpu")),
            "--hidden-dim", str(hidden_dim),
            "--n-hidden", str(n_hidden),
            "--bc-target-margin", str(target_margin),
            "--linear-init-margin", str(target_margin),
            "--bc-margin-mode", "all",
            "--rashomon-multi-label-mode", "all",
            "--rashomon-surrogate", ADAPTIVE_SURROGATE,
            "--rashomon-batch-size", str(dataset_size),
            "--certificate-samples", str(dataset_size),
        ]
        print(f"preparing all-safe base policy for {environment}", flush=True)
        subprocess.check_call(command, cwd=str(REPO))
    return base_dir / "base_policy.pt"


def build_jobs() -> list[dict]:
    if ADAPTIVE_VERSION not in {"v1", "v2"}:
        raise SystemExit(f"ADAPTIVE_VERSION must be v1 or v2, got {ADAPTIVE_VERSION!r}")
    if ADAPTIVE_TOTAL_ITERS is not None:
        raise SystemExit(
            "ADAPTIVE_TOTAL_ITERS is legacy-only; use ADAPTIVE_N_ITERS as the "
            "per-computation budget"
        )
    jobs: list[dict] = []
    for env in ENVS:
        cfg = _resolve_env_settings(ENV_PIPELINE[env])
        base_policy = _canonical_base_dir(env) / "base_policy.pt"
        if not base_policy.exists():
            raise SystemExit(f"{env}: missing safe base policy {base_policy}")
        total_timesteps = SMOKE or cfg["total_timesteps"]
        for seed in SEEDS:
            run_dir = f"{OUT_BASE}/{env}"
            cmd = [
                PY, STAGE,
                "--base-policy-path", str(base_policy),
                "--shield-path", str(cfg["shield_path"]),
                "--env-id", cfg["env_id"],
                "--env-kwargs", json.dumps(cfg["env_kwargs"] or {}),
                "--state-representation", "one_hot",
                "--max-episode-steps", str(cfg["max_episode_steps"]),
                "--cost-limit", str(cfg["cost_limit"]),
                "--total-timesteps", str(total_timesteps),
                "--eval-episodes", str(cfg["eval_episodes"]),
                "--seed", str(seed),
                "--learning-rate", str(cfg["learning_rate"]),
                "--n-steps", str(cfg["n_steps"]),
                "--batch-size", str(cfg["batch_size"]),
                "--n-epochs", str(cfg["n_epochs"]),
                "--gamma", str(cfg["gamma"]),
                "--gae-lambda", str(cfg["gae_lambda"]),
                "--clip-range", str(cfg["clip_range"]),
                "--ent-coef", str(cfg["ent_coef"]),
                "--vf-coef", str(cfg["vf_coef"]),
                "--max-grad-norm", str(cfg["max_grad_norm"]),
                "--device", str(cfg.get("device", "cpu")),
                "--early-stop-eval-policy", "unshielded",
                "--evaluation-policy", str(cfg.get("rashomon_evaluation_policy", "unshielded")),
                "--early-stop-eval-freq", str(cfg["early_stop_eval_freq"]),
                "--early-stop-eval-episodes", str(cfg["early_stop_eval_episodes"]),
                "--early-stop-success-rate", str(cfg["early_stop_success_rate"]),
                "--success-reward-threshold", str(cfg["success_reward_threshold"]),
                "--curve-eval-freq", str(cfg["curve_eval_freq"]),
                "--curve-eval-episodes", str(cfg["curve_eval_episodes"]),
                "--output-dir", run_dir,
                "--run-id", f"seed{seed}",
                "--verify-first", ADAPTIVE_VERIFY_FIRST,
                "--freq", ADAPTIVE_FREQ,
                "--directional", ADAPTIVE_DIRECTIONAL,
                "--region-mode", ADAPTIVE_REGION_MODE,
                "--n-iters", ADAPTIVE_N_ITERS,
                "--rashomon-multi-label-mode", "all",
                "--surrogate", ADAPTIVE_SURROGATE,
            ]
            if ADAPTIVE_STRATEGY != "rashomon_project":
                cmd.extend(["--unsafe-update-strategy", ADAPTIVE_STRATEGY])
            jobs.append({"tag": f"{env}/seed{seed}", "cmd": cmd})
    return jobs


def _assign_cores(n_jobs: int) -> list[int | None]:
    """One distinct CPU core id per job (None entries disable pinning)."""
    if NO_PIN:
        return [None] * n_jobs
    cpu_ids = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    # CPU_OFFSET lets this launcher share the machine with a concurrent sweep
    # (e.g. run_seed_experiments.py) without both starting at core 0 and
    # colliding, which silently halves throughput.
    offset = int(os.environ.get("CPU_OFFSET", "0"))
    if len(cpu_ids) < n_jobs:
        print(
            f"warning: only {len(cpu_ids)} cores available for {n_jobs} jobs; "
            "reusing cores round-robin.",
            flush=True,
        )
    if not cpu_ids:
        return [None] * n_jobs
    return [cpu_ids[(offset + i) % len(cpu_ids)] for i in range(n_jobs)]


def run_job(job: dict) -> tuple[str, int, float]:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / (job["tag"].replace("/", "__") + ".log")
    cmd = job["cmd"]
    if job.get("core") is not None:
        cmd = ["taskset", "-c", str(job["core"]), *cmd]
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return job["tag"], rc, time.time() - t0


def main() -> int:
    for environment in ENVS:
        _prepare_all_safe_base_policy(
            environment, _resolve_env_settings(ENV_PIPELINE[environment])
        )
    jobs = build_jobs()
    cores = _assign_cores(len(jobs))
    for job, core in zip(jobs, cores):
        job["core"] = core
    print(
        f"launching {len(jobs)} AdaptiveSafePPO jobs (envs={len(ENVS)} x seeds={len(SEEDS)}), "
        f"verify_first={ADAPTIVE_VERIFY_FIRST}, strategy={ADAPTIVE_STRATEGY}, "
        f"n_iters={ADAPTIVE_N_ITERS}, freq={ADAPTIVE_FREQ}, "
        f"directional={ADAPTIVE_DIRECTIONAL}, region_mode={ADAPTIVE_REGION_MODE}, "
        f"pinning={'off' if NO_PIN else 'one core per job'}, smoke={SMOKE or 'off'}",
        flush=True,
    )
    done = fail = 0
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(run_job, j): j["tag"] for j in jobs}
        for fut in as_completed(futs):
            tag, rc, secs = fut.result()
            done += 1
            fail += int(rc != 0)
            print(f"[{done}/{len(jobs)}] {'ok' if rc == 0 else f'FAIL(rc={rc})'} {tag} ({secs:.0f}s)", flush=True)
    print(f"ADAPTIVE_SWEEP_DONE total={len(jobs)} failed={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
