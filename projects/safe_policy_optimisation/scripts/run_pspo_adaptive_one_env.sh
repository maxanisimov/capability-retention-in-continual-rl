#!/usr/bin/env bash
set -euo pipefail

cd /vol/bitbucket/ma5923/_projects/CertifiedContinualLearning

: "${ENV_NAME:?set ENV_NAME}"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CORE_START="${CORE_START:-}"
CPU_IDS="${CPU_IDS:-}"
REGION_MODE="${REGION_MODE:-replace}"
RUN_NAME="${RUN_NAME:-pspo_adaptive_tabular_best}"
ARCHITECTURE="${ARCHITECTURE:-tabular}"
RASHOMON_N_ITERS="${RASHOMON_N_ITERS:-}"
RASHOMON_MULTI_LABEL_MODE="${RASHOMON_MULTI_LABEL_MODE:-all}"
RASHOMON_SURROGATE="${RASHOMON_SURROGATE:-logsumexp}"
RASHOMON_BATCH_SIZE="${RASHOMON_BATCH_SIZE:-}"
RASHOMON_CERTIFICATE_SAMPLES="${RASHOMON_CERTIFICATE_SAMPLES:-}"
BC_TARGET_MARGIN="${BC_TARGET_MARGIN:-}"
DIRECTIONAL_RASHOMON_GROWTH="${DIRECTIONAL_RASHOMON_GROWTH:-1}"
ADAPTIVE_GRANULARITY="${ADAPTIVE_GRANULARITY:-}"
ADAPTIVE_FREQ="${ADAPTIVE_FREQ:-}"
STOP_WHEN_PROPOSAL_CONTAINED="${STOP_WHEN_PROPOSAL_CONTAINED:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$CPU_IDS" && -z "$CORE_START" ]]; then
  echo "Set CPU_IDS (explicit cores) or CORE_START (legacy consecutive allocation)." >&2
  exit 2
fi

export ENV_NAME CORE_START CPU_IDS SEEDS REGION_MODE RUN_NAME ARCHITECTURE RASHOMON_N_ITERS
export RASHOMON_MULTI_LABEL_MODE RASHOMON_SURROGATE RASHOMON_BATCH_SIZE
export RASHOMON_CERTIFICATE_SAMPLES
export BC_TARGET_MARGIN DRY_RUN
export DIRECTIONAL_RASHOMON_GROWTH
export ADAPTIVE_GRANULARITY
export ADAPTIVE_FREQ
export STOP_WHEN_PROPOSAL_CONTAINED
export SKIP_EXISTING

.venv/bin/python - <<'PY'
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings
from projects.safe_policy_optimisation.utils.pspo_adaptive_launcher import (
    base_policy_artifact_matches,
    resolve_certificate_samples,
    resolve_seed_cpu_ids,
    resolve_target_margin,
)

REPO = Path.cwd()
ENV_NAME = os.environ["ENV_NAME"]
SEEDS = [int(x) for x in os.environ["SEEDS"].split()]
try:
    CPU_IDS = resolve_seed_cpu_ids(
        SEEDS,
        cpu_ids=os.environ["CPU_IDS"],
        core_start=os.environ["CORE_START"],
    )
except ValueError as exc:
    raise SystemExit(f"Invalid CPU allocation: {exc}") from exc
unavailable_cpus = sorted(set(CPU_IDS) - set(os.sched_getaffinity(0)))
if unavailable_cpus:
    raise SystemExit(f"Requested CPUs are outside this process's affinity: {unavailable_cpus}")
REGION_MODE = os.environ["REGION_MODE"]
RUN_NAME = os.environ["RUN_NAME"]
ARCHITECTURE = os.environ["ARCHITECTURE"]
RASHOMON_N_ITERS = os.environ["RASHOMON_N_ITERS"]
RASHOMON_MULTI_LABEL_MODE = os.environ["RASHOMON_MULTI_LABEL_MODE"]
RASHOMON_SURROGATE = os.environ["RASHOMON_SURROGATE"]
RASHOMON_BATCH_SIZE = os.environ["RASHOMON_BATCH_SIZE"]
RASHOMON_CERTIFICATE_SAMPLES = os.environ["RASHOMON_CERTIFICATE_SAMPLES"]
BC_TARGET_MARGIN = os.environ["BC_TARGET_MARGIN"]
DIRECTIONAL_RASHOMON_GROWTH = os.environ["DIRECTIONAL_RASHOMON_GROWTH"] == "1"
ADAPTIVE_GRANULARITY = os.environ["ADAPTIVE_GRANULARITY"]
ADAPTIVE_FREQ = os.environ["ADAPTIVE_FREQ"]
STOP_WHEN_PROPOSAL_CONTAINED = os.environ["STOP_WHEN_PROPOSAL_CONTAINED"] == "1"
SKIP_EXISTING = os.environ["SKIP_EXISTING"] == "1"
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
if RASHOMON_MULTI_LABEL_MODE not in {"any", "all"}:
    raise SystemExit("RASHOMON_MULTI_LABEL_MODE must be 'any' or 'all'")
if RASHOMON_SURROGATE not in {"auto", "probability", "logsumexp"}:
    raise SystemExit(
        "RASHOMON_SURROGATE must be 'auto', 'probability', or 'logsumexp'"
    )
if ADAPTIVE_GRANULARITY and ADAPTIVE_GRANULARITY not in {"gradient_step", "train_phase"}:
    raise SystemExit("ADAPTIVE_GRANULARITY must be 'gradient_step' or 'train_phase'")
if not ADAPTIVE_FREQ:
    ADAPTIVE_FREQ = "rollout" if ADAPTIVE_GRANULARITY == "train_phase" else "update"

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


try:
    bc_target_margin = resolve_target_margin(
        BC_TARGET_MARGIN,
        default=float(hp.get("bc_target_margin", 10.0)),
    )
except ValueError as exc:
    raise SystemExit(f"Invalid BC_TARGET_MARGIN: {exc}") from exc

from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    load_shield_mask,
    make_safe_behaviour_payload,
)

shield_path = repo_path(cfg["shield_path"])
shield_mask = load_shield_mask(shield_path)
_, safety_dataset_metadata = make_safe_behaviour_payload(shield_mask)
safety_demo_size = int(safety_dataset_metadata["dataset_size"])
try:
    certificate_samples = resolve_certificate_samples(
        RASHOMON_CERTIFICATE_SAMPLES,
        default=int(hp.get("certificate_samples", 1000)),
        shield_mask=shield_mask,
    )
except ValueError as exc:
    raise SystemExit(f"Invalid RASHOMON_CERTIFICATE_SAMPLES: {exc}") from exc

if RASHOMON_BATCH_SIZE == "all":
    rashomon_batch_size = safety_demo_size
elif RASHOMON_BATCH_SIZE:
    try:
        rashomon_batch_size = int(RASHOMON_BATCH_SIZE)
    except ValueError as exc:
        raise SystemExit(
            "RASHOMON_BATCH_SIZE must be a positive integer or 'all'"
        ) from exc
else:
    rashomon_batch_size = int(hp.get("rashomon_batch_size", 500))
if rashomon_batch_size <= 0:
    raise SystemExit("RASHOMON_BATCH_SIZE must be positive")

if RASHOMON_BATCH_SIZE == "all" and rashomon_batch_size != safety_demo_size:
    raise AssertionError("full Rashomon batch does not match the safety dataset")
if RASHOMON_CERTIFICATE_SAMPLES == "all" and certificate_samples != safety_demo_size:
    raise AssertionError("full certificate coverage does not match the safety dataset")


def base_policy_command(base_dir):
    return [
        str(REPO / ".venv/bin/python"),
        str(REPO / "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py"),
        "--output-dir",
        str(base_dir.parent),
        "--run-id",
        base_dir.name,
        "--base-policy-only",
        "--shield-path",
        str(shield_path),
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
        str(bc_target_margin),
        "--linear-init-margin",
        str(bc_target_margin),
        "--bc-margin-mode",
        RASHOMON_MULTI_LABEL_MODE,
        "--rashomon-multi-label-mode",
        RASHOMON_MULTI_LABEL_MODE,
        "--rashomon-surrogate",
        RASHOMON_SURROGATE,
        "--rashomon-batch-size",
        str(rashomon_batch_size),
        "--certificate-samples",
        str(certificate_samples),
    ]


if RASHOMON_MULTI_LABEL_MODE == "all":
    base_dir = out_base / "initial_base_policy"
    base_policy = base_dir / "base_policy.pt"
    if not base_policy_artifact_matches(
        base_dir,
        shield_path=shield_path,
        dataset_size=safety_demo_size,
        hidden_dim=int(hp.get("hidden_dim", 64)),
        n_hidden=int(hp.get("n_hidden", 0 if ARCHITECTURE == "tabular" else 2)),
        state_representation="one_hot_discrete_observation",
        margin_mode=RASHOMON_MULTI_LABEL_MODE,
        target_margin=bc_target_margin,
    ):
        set_cmd = base_policy_command(base_dir)
        if DRY_RUN:
            print(f"dry-run base-policy: {' '.join(set_cmd)}", flush=True)
        else:
            log = logdir / "initial_base_policy.log"
            with log.open("w") as fh:
                rc = subprocess.run(
                    ["taskset", "-c", str(CPU_IDS[0]), *set_cmd],
                    cwd=REPO,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                ).returncode
            if rc != 0:
                raise SystemExit(f"Base-policy preparation failed with rc={rc}; see {log}")
else:
    base_policy = repo_path(hp["rashomon_dir"]) / "base_policy.pt"

if not base_policy.exists() and not DRY_RUN:
    raise SystemExit(f"Missing base policy: {base_policy}")

jobs = []
for seed, core in zip(SEEDS, CPU_IDS):
    completed_metrics = out_base / f"seed{seed}" / "metrics.json"
    if SKIP_EXISTING and completed_metrics.exists():
        print(f"{ENV_NAME}: skipping completed seed{seed}: {completed_metrics}", flush=True)
        continue
    cmd = [
        str(REPO / ".venv/bin/python"),
        str(REPO / "projects/safe_policy_optimisation/stages/train_pspo_adaptive.py"),
        "--base-policy-path",
        str(base_policy),
        "--shield-path",
        str(shield_path),
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
        "--verify-first",
        "false",
        "--freq",
        ADAPTIVE_FREQ,
        "--directional",
        "true" if DIRECTIONAL_RASHOMON_GROWTH else "false",
        "--region-mode",
        REGION_MODE,
        "--rashomon-checkpoint",
        str(hp.get("checkpoint", 100)),
        "--rashomon-batch-size",
        str(rashomon_batch_size),
        "--certificate-samples",
        str(certificate_samples),
        "--rashomon-multi-label-mode",
        RASHOMON_MULTI_LABEL_MODE,
        "--surrogate",
        RASHOMON_SURROGATE,
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
    n_iters = int(RASHOMON_N_ITERS) if RASHOMON_N_ITERS else int(hp["rashomon_n_iters"])
    if n_iters <= 0:
        raise SystemExit("RASHOMON_N_ITERS must be positive")
    cmd[cmd.index("--rashomon-checkpoint"):cmd.index("--rashomon-checkpoint")] = [
        "--n-iters",
        str(n_iters),
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
print(
    f"{ENV_NAME}: safety demonstrations={safety_demo_size}, "
    f"rashomon_batch_size={rashomon_batch_size}, "
    f"certificate_samples={certificate_samples}",
    flush=True,
)

if DRY_RUN:
    for seed, core, cmd in jobs:
        print(f"dry-run seed{seed} core={core}: {' '.join(cmd)}", flush=True)
    raise SystemExit(0)

if not jobs:
    print(f"{ENV_NAME}: all requested seeds are already complete", flush=True)
    raise SystemExit(0)

with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
    futures = [ex.submit(run, job) for job in jobs]
    for fut in as_completed(futures):
        seed, core, rc, secs = fut.result()
        status = "ok" if rc == 0 else f"FAIL rc={rc}"
        print(f"{status} seed{seed} core={core} {secs:.0f}s", flush=True)
PY
