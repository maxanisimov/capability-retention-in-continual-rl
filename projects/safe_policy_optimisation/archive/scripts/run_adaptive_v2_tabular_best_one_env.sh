#!/usr/bin/env bash
set -euo pipefail

cd /vol/bitbucket/ma5923/_projects/CertifiedContinualLearning

: "${ENV_NAME:?set ENV_NAME}"
: "${CORE_START:?set CORE_START}"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
REGION_MODE="${REGION_MODE:-union}"
RUN_NAME="${RUN_NAME:-adaptive_v2_tabular_best_precomputed}"
ARCHITECTURE="${ARCHITECTURE:-tabular}"
RASHOMON_TOTAL_ITERS="${RASHOMON_TOTAL_ITERS:-}"
RASHOMON_INITIAL_N_ITERS="${RASHOMON_INITIAL_N_ITERS:-}"
RASHOMON_RECOMPUTE_N_ITERS="${RASHOMON_RECOMPUTE_N_ITERS:-}"
INITIAL_SET_RASHOMON_N_ITERS="${INITIAL_SET_RASHOMON_N_ITERS:-}"
RASHOMON_MULTI_LABEL_MODE="${RASHOMON_MULTI_LABEL_MODE:-any}"
DRY_RUN="${DRY_RUN:-0}"

export ENV_NAME CORE_START SEEDS REGION_MODE RUN_NAME ARCHITECTURE RASHOMON_TOTAL_ITERS
export RASHOMON_INITIAL_N_ITERS RASHOMON_RECOMPUTE_N_ITERS INITIAL_SET_RASHOMON_N_ITERS
export RASHOMON_MULTI_LABEL_MODE DRY_RUN

python - <<'PY'
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings

REPO = Path.cwd()
ENV_NAME = os.environ["ENV_NAME"]
CORE_START = int(os.environ["CORE_START"])
SEEDS = [int(x) for x in os.environ["SEEDS"].split()]
REGION_MODE = os.environ["REGION_MODE"]
RUN_NAME = os.environ["RUN_NAME"]
ARCHITECTURE = os.environ["ARCHITECTURE"]
RASHOMON_TOTAL_ITERS = os.environ["RASHOMON_TOTAL_ITERS"]
RASHOMON_INITIAL_N_ITERS = os.environ["RASHOMON_INITIAL_N_ITERS"]
RASHOMON_RECOMPUTE_N_ITERS = os.environ["RASHOMON_RECOMPUTE_N_ITERS"]
INITIAL_SET_RASHOMON_N_ITERS = os.environ["INITIAL_SET_RASHOMON_N_ITERS"]
RASHOMON_MULTI_LABEL_MODE = os.environ["RASHOMON_MULTI_LABEL_MODE"]
DRY_RUN = os.environ["DRY_RUN"] == "1"

PIPELINES = {
    "media_streaming": ("paper_2503_07671_media_streaming", "Media Streaming"),
    "colour_bomb": ("paper_2503_07671_colour_bomb", "Colour Bomb v1"),
    "colour_bomb_v2": ("paper_2503_07671_colour_bomb_v2", "Colour Bomb v2"),
    "bridge_crossing": ("paper_2503_07671_bridge_crossing", "Bridge Crossing v1"),
    "bridge_crossing_v2": ("paper_2503_07671_bridge_crossing_v2", "Bridge Crossing v2"),
    "mini_pacman": ("paper_2503_07671_minipacman", "MiniPacman"),
}

if ENV_NAME not in PIPELINES:
    choices = ", ".join(sorted(PIPELINES))
    raise SystemExit(f"Unknown ENV_NAME={ENV_NAME!r}; expected one of: {choices}")

pipeline, label = PIPELINES[ENV_NAME]
cfg, _, _ = compose_pipeline_settings(pipeline)
best_path = (
    REPO
    / "projects/safe_policy_optimisation/docs/pspo_precomputed"
    / "pspo_precomputed_best_hyperparameters.json"
)
best = json.loads(best_path.read_text())
alt_best_path = (
    REPO
    / "projects/safe_policy_optimisation/docs/pspo_precomputed"
    / "best_pspo_precomputed_by_architecture_current.json"
)


def best_hyperparameters(architecture, env_label):
    try:
        hp = best["architectures"][architecture][env_label]["best_settings"][0]["hyperparameters"]
        return dict(hp), best_path
    except KeyError:
        pass
    alt_best = json.loads(alt_best_path.read_text())
    try:
        payload = alt_best["architectures"][architecture][env_label]
        hp = payload.get("hyperparameters")
        if hp is None:
            hp = payload["best_settings"][0]["hyperparameters"]
    except KeyError as exc:
        available = sorted(
            set(best.get("architectures", {}).get(architecture, {}))
            | set(alt_best.get("architectures", {}).get(architecture, {}))
        )
        raise SystemExit(
            f"No best precomputed PSPO hyperparameters for ARCHITECTURE={architecture!r}, "
            f"ENV_NAME={ENV_NAME!r} ({env_label!r}). Available labels: {available}"
        ) from exc
    hp = dict(hp)
    if architecture == "one_hidden":
        hp.setdefault("n_hidden", 1)
        hp.setdefault("hidden_dim", 64)
    elif architecture == "tabular":
        hp.setdefault("n_hidden", 0)
        hp.setdefault("hidden_dim", 64)
    return hp, alt_best_path


hp, hp_source = best_hyperparameters(ARCHITECTURE, label)

out_base = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
    / RUN_NAME
)
if ARCHITECTURE == "tabular":
    out_base = out_base / ENV_NAME
else:
    out_base = out_base / ARCHITECTURE / ENV_NAME
logdir = out_base / "_launch_logs"
logdir.mkdir(parents=True, exist_ok=True)


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def initial_set_matches(safe_set_dir):
    bounds_path = safe_set_dir / "rashomon_param_bounds.pt"
    base_path = safe_set_dir / "base_policy.pt"
    summary_path = safe_set_dir / "summary.json"
    if not bounds_path.exists() or not base_path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(summary, dict):
        return False
    base_policy_summary = summary.get("base_policy") or {}
    rashomon_summary = summary.get("rashomon") or {}
    if not isinstance(base_policy_summary, dict) or not isinstance(rashomon_summary, dict):
        return False
    return (
        base_policy_summary.get("bc_margin_mode") == RASHOMON_MULTI_LABEL_MODE
        and rashomon_summary.get("multi_label_mode") == RASHOMON_MULTI_LABEL_MODE
    )


def initial_set_command(safe_set_dir):
    n_iters = INITIAL_SET_RASHOMON_N_ITERS or hp.get("rashomon_n_iters", 2000)
    return [
        str(REPO / ".venv/bin/python"),
        str(REPO / "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py"),
        "--output-dir",
        str(safe_set_dir.parent),
        "--run-id",
        safe_set_dir.name,
        "--shield-path",
        str(repo_path(cfg["shield_path"])),
        "--env-id",
        cfg["env_id"],
        "--env-kwargs",
        json.dumps(cfg.get("env_kwargs") or {}),
        "--state-representation",
        "one_hot",
        "--seed",
        "0",
        "--device",
        str(cfg.get("device", "cpu")),
        "--hidden-dim",
        str(hp.get("hidden_dim", 64)),
        "--n-hidden",
        str(hp.get("n_hidden", 0 if ARCHITECTURE == "tabular" else 2)),
        "--bc-target-margin",
        str(hp.get("bc_target_margin", 10.0)),
        "--linear-init-margin",
        str(hp.get("bc_target_margin", 10.0)),
        "--bc-margin-mode",
        RASHOMON_MULTI_LABEL_MODE,
        "--rashomon-multi-label-mode",
        RASHOMON_MULTI_LABEL_MODE,
        "--rashomon-n-iters",
        str(int(n_iters)),
        "--rashomon-checkpoint",
        str(hp.get("checkpoint", hp.get("rashomon_checkpoint", 100))),
        "--rashomon-batch-size",
        str(hp.get("rashomon_batch_size", 500)),
        "--certificate-samples",
        str(hp.get("certificate_samples", 1000)),
    ]


if RASHOMON_MULTI_LABEL_MODE == "all":
    safe_set_dir = out_base / "initial_safe_set"
    base_policy = safe_set_dir / "base_policy.pt"
    if not initial_set_matches(safe_set_dir):
        set_cmd = initial_set_command(safe_set_dir)
        if DRY_RUN:
            print(f"dry-run initial-safe-set: {' '.join(set_cmd)}", flush=True)
        else:
            log = logdir / "initial_safe_set.log"
            with log.open("w") as fh:
                rc = subprocess.run(
                    set_cmd,
                    cwd=REPO,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                ).returncode
            if rc != 0:
                raise SystemExit(f"Initial safe-set computation failed with rc={rc}; see {log}")
else:
    base_policy = repo_path(hp["rashomon_dir"]) / "base_policy.pt"

if not base_policy.exists() and not DRY_RUN:
    raise SystemExit(f"Missing base policy: {base_policy}")

jobs = []
for i, seed in enumerate(SEEDS):
    core = CORE_START + i
    cmd = [
        str(REPO / ".venv/bin/python"),
        str(REPO / "projects/safe_policy_optimisation/stages/train_pspo_adaptive_v2.py"),
        "--base-policy-path",
        str(base_policy),
        "--shield-path",
        str(REPO / cfg["shield_path"]),
        "--env-id",
        cfg["env_id"],
        "--env-kwargs",
        json.dumps(cfg.get("env_kwargs") or {}),
        "--state-representation",
        "one_hot",
        "--max-episode-steps",
        str(cfg["max_episode_steps"]),
        "--cost-limit",
        str(cfg["cost_limit"]),
        "--total-timesteps",
        str(cfg["total_timesteps"]),
        "--eval-episodes",
        str(cfg["eval_episodes"]),
        "--seed",
        str(seed),
        "--region-update-mode",
        REGION_MODE,
        "--rashomon-checkpoint",
        str(hp.get("checkpoint", 100)),
        "--rashomon-batch-size",
        str(hp.get("rashomon_batch_size", 500)),
        "--certificate-samples",
        str(hp.get("certificate_samples", 1000)),
        "--rashomon-multi-label-mode",
        RASHOMON_MULTI_LABEL_MODE,
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
        str(cfg.get("device", "cpu")),
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
        "--output-dir",
        str(out_base),
        "--run-id",
        f"seed{seed}",
    ]
    budget_args = []
    if RASHOMON_TOTAL_ITERS:
        budget_args.extend(["--rashomon-total-iters", str(int(RASHOMON_TOTAL_ITERS))])
    if RASHOMON_INITIAL_N_ITERS:
        budget_args.extend(["--rashomon-initial-n-iters", str(int(RASHOMON_INITIAL_N_ITERS))])
    if RASHOMON_RECOMPUTE_N_ITERS:
        budget_args.extend(["--rashomon-recompute-n-iters", str(int(RASHOMON_RECOMPUTE_N_ITERS))])
    if budget_args:
        cmd[cmd.index("--rashomon-checkpoint"):cmd.index("--rashomon-checkpoint")] = [
            *budget_args,
        ]
    else:
        cmd[cmd.index("--rashomon-checkpoint"):cmd.index("--rashomon-checkpoint")] = [
            "--rashomon-n-iters",
            str(hp["rashomon_n_iters"]),
        ]
    jobs.append((seed, core, cmd))


def run(job):
    seed, core, cmd = job
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    log = logdir / f"seed{seed}.log"
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(
            ["taskset", "-c", str(core), *cmd],
            cwd=REPO,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        ).returncode
    return seed, core, rc, time.time() - t0


print(f"{ENV_NAME}: launching {len(jobs)} seeds on cores {[j[1] for j in jobs]}", flush=True)
print(f"{ENV_NAME}: base policy {base_policy}", flush=True)
print(f"{ENV_NAME}: output {out_base}", flush=True)

if DRY_RUN:
    for seed, core, cmd in jobs:
        print(f"dry-run seed{seed} core={core}: {' '.join(cmd)}", flush=True)
    raise SystemExit(0)

with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
    futures = [ex.submit(run, job) for job in jobs]
    for fut in as_completed(futures):
        seed, core, rc, secs = fut.result()
        status = "ok" if rc == 0 else f"FAIL rc={rc}"
        print(f"{status} seed{seed} core={core} {secs:.0f}s", flush=True)
PY
