#!/usr/bin/env bash
# Launch PSPO on LunarLander-v3 with the loose descent shield, one seed per
# idle CPU core, each in its own detached screen session.
#
# The shield admits only the main engine inside y < 0.6 and v_y < -0.35; the
# unsafe set it protects is the stricter y < 0.5 and v_y < -0.4, so the shield
# acts before a violation rather than on top of one. Hyperparameters match the
# vanilla PPO LunarLander baseline in
# outputs/continuous_state_shields/ppo_baselines/20260907_212347_10seed_three_env
# so the two are directly comparable.
#
# Usage:
#   projects/safe_policy_optimisation/scripts/run_pspo_lunarlander_descent.sh
#   SEEDS="0 1" CPU_IDS="10,11" DRY_RUN=1 ...same script...
#
# Segment arm of the orthotope-vs-segment A/B. CERT_ROOT and INIT_ROOT are
# pinned to the 2026-09-09 run so both arms start from the same certificate and
# the same 10 base policies; only RUN_ID (and so RUN_ROOT) changes, because
# those three roots otherwise all derive from RUN_ID and would be rebuilt:
#   RUN_ID=20260916_lunarlander_descent_segment \
#   SAFE_REGION_SHAPE=segment \
#   SESSION_PREFIX=pspo_ll_seg \
#   CERT_ROOT=$REPO/outputs/continuous_state_shields/synthesised/lunarlander_descent/20260909_lunarlander_descent_loose \
#   INIT_ROOT=$REPO/outputs/continuous_state_shields/pspo_initialisation/20260909_lunarlander_descent_loose \
#   projects/safe_policy_optimisation/scripts/run_pspo_lunarlander_descent.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
RUN_ID="${RUN_ID:-20260909_lunarlander_descent_loose}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/outputs/continuous_state_shields/pspo/$RUN_ID}"
CERT_ROOT="${CERT_ROOT:-$REPO_DIR/outputs/continuous_state_shields/synthesised/lunarlander_descent/$RUN_ID}"
INIT_ROOT="${INIT_ROOT:-$REPO_DIR/outputs/continuous_state_shields/pspo_initialisation/$RUN_ID}"

# Shield band (activation) and the stricter unsafe set it protects.
CRITICAL_HEIGHT="${CRITICAL_HEIGHT:-0.6}"
SAFE_MIN_VERTICAL_SPEED="${SAFE_MIN_VERTICAL_SPEED:--0.35}"
UNSAFE_HEIGHT="${UNSAFE_HEIGHT:-0.5}"
UNSAFE_VERTICAL_SPEED="${UNSAFE_VERTICAL_SPEED:--0.4}"

# PSPO settings: canonical region-first, directional, replace, LSE surrogate.
VERIFY_FIRST="${VERIFY_FIRST:-false}"
DIRECTIONAL="${DIRECTIONAL:-true}"
REGION_MODE="${REGION_MODE:-replace}"
RASHOMON_SURROGATE="${RASHOMON_SURROGATE:-logsumexp}"
RASHOMON_N_ITERS="${RASHOMON_N_ITERS:-200}"
RASHOMON_MULTI_LABEL_MODE="${RASHOMON_MULTI_LABEL_MODE:-all}"
FREQ="${FREQ:-1}"
# Certified-region geometry. 'orthotope' clips the proposal onto a box, which on
# continuous states projects every single update and destroys reward; 'segment'
# certifies the longest admissible step along the proposal, so the direction
# survives and only the magnitude shrinks. Segment requires --freq != once,
# --region-refresh adaptive, --region-mode != union, and GROWTH_METHOD=IBP (the
# segment engine ignores any other growth verifier), all enforced at parse time.
SAFE_REGION_SHAPE="${SAFE_REGION_SHAPE:-orthotope}"
SEGMENT_TOLERANCE="${SEGMENT_TOLERANCE:-0.001}"
SEGMENT_SPLITS="${SEGMENT_SPLITS:-4}"
SEGMENT_MAX_SPLITS="${SEGMENT_MAX_SPLITS:-8}"
# Verifier used to grow and to accept certified regions, and to state the final
# certificate. IBP is the cheap default; CROWN is tighter but ~10x slower per
# region, and alpha-CROWN is ~200x and impractical inside the training loop.
GROWTH_METHOD="${GROWTH_METHOD:-IBP}"
CERTIFICATION_METHOD="${CERTIFICATION_METHOD:-IBP}"
# The base policy must be certified under the same verifier PSPO will check it
# with, or the run aborts on its initial safety check.
INIT_CERTIFICATION_METHOD="${INIT_CERTIFICATION_METHOD:-$CERTIFICATION_METHOD}"

# PPO budget and hyperparameters, matched to the vanilla PPO baseline.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_EPOCHS="${N_EPOCHS:-10}"
ENT_COEF="${ENT_COEF:-0.01}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
SUCCESS_REWARD_THRESHOLD="${SUCCESS_REWARD_THRESHOLD:-200}"

# Base-policy initialisation.
HIDDEN_DIM="${HIDDEN_DIM:-64}"
N_HIDDEN="${N_HIDDEN:-2}"
BC_SAMPLES="${BC_SAMPLES:-4096}"
BC_EPOCHS="${BC_EPOCHS:-200}"
INIT_MAX_EPOCHS="${INIT_MAX_EPOCHS:-2000}"
INIT_TARGET_MARGIN="${INIT_TARGET_MARGIN:-2.0}"

CPU_IDS="${CPU_IDS:-}"
MINIMUM_IDLE="${MINIMUM_IDLE:-90}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

PYTHON="$REPO_DIR/.venv/bin/python"
export PYTHONPATH="$REPO_DIR/core:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy

read -r -a SEED_ARRAY <<< "$SEEDS"
n_seeds="${#SEED_ARRAY[@]}"

# ---------------------------------------------------------------- CPU cores --
# One core per seed. Without an explicit list, pick the most idle cores inside
# this process's affinity mask so the run never lands on a busy core.
if [[ -n "$CPU_IDS" ]]; then
    IFS=',' read -r -a CORE_ARRAY <<< "$CPU_IDS"
else
    echo "sampling per-core idle time for ${SAMPLE_SECONDS}s ..."
    mapfile -t CORE_ARRAY < <(
        "$PYTHON" - "$n_seeds" "$MINIMUM_IDLE" "$SAMPLE_SECONDS" <<'PY'
import os
import subprocess
import sys

from projects.safe_policy_optimisation.scripts.launch_pspo_multi_env import (
    parse_mpstat_idle,
    select_idle_cpus,
)

required, minimum_idle, sample_seconds = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
completed = subprocess.run(
    ["mpstat", "-P", "ALL", str(sample_seconds), "1"],
    check=True,
    capture_output=True,
    text=True,
)
idle = parse_mpstat_idle(completed.stdout)
for cpu in select_idle_cpus(
    idle,
    required=required,
    minimum_idle=minimum_idle,
    allowed_cpus=set(os.sched_getaffinity(0)),
):
    print(cpu)
PY
    )
fi

if [[ "${#CORE_ARRAY[@]}" -ne "$n_seeds" ]]; then
    echo "need exactly $n_seeds cores for $n_seeds seeds; got ${#CORE_ARRAY[@]}: ${CORE_ARRAY[*]}" >&2
    exit 2
fi
echo "seeds:  ${SEED_ARRAY[*]}"
echo "cores:  ${CORE_ARRAY[*]}"

# --------------------------------------------------------------- certificate --
cert_dir="$CERT_ROOT/descent_band"
cert_path="$cert_dir/critical_interval_dataset.pt"
if [[ -f "$cert_path" && "$SKIP_EXISTING" == "1" ]]; then
    echo "certificate: reusing $cert_path"
else
    echo "certificate: building $cert_path"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$PYTHON" "$REPO_DIR/projects/safe_policy_optimisation/scripts/build_lunarlander_descent_certificate.py" \
            --output-dir "$CERT_ROOT" \
            --run-id descent_band \
            --critical-height "$CRITICAL_HEIGHT" \
            --safe-min-vertical-speed "$SAFE_MIN_VERTICAL_SPEED"
    fi
fi

# ------------------------------------------------------------- base policies --
# One certified base policy per seed, so PSPO's starting point varies with the
# seed exactly as the PPO baseline's initialisation does.
for index in "${!SEED_ARRAY[@]}"; do
    seed="${SEED_ARRAY[$index]}"
    core="${CORE_ARRAY[$index]}"
    base_policy="$INIT_ROOT/seed_$seed/base_policy.pt"
    if [[ -f "$base_policy" && "$SKIP_EXISTING" == "1" ]]; then
        echo "base policy seed $seed: reusing $base_policy"
        continue
    fi
    echo "base policy seed $seed: training on core $core"
    [[ "$DRY_RUN" == "1" ]] && continue
    taskset -c "$core" "$PYTHON" \
        "$REPO_DIR/projects/safe_policy_optimisation/stages/train_mountaincar_pspo_initialisation.py" \
        --output-dir "$INIT_ROOT" \
        --run-id "seed_$seed" \
        --obs-dim 8 \
        --n-actions 4 \
        --certificate-dataset "$cert_path" \
        --hidden-dim "$HIDDEN_DIM" \
        --n-hidden "$N_HIDDEN" \
        --bc-samples "$BC_SAMPLES" \
        --bc-epochs "$BC_EPOCHS" \
        --max-epochs "$INIT_MAX_EPOCHS" \
        --target-margin "$INIT_TARGET_MARGIN" \
        --certification-method "$INIT_CERTIFICATION_METHOD" \
        --seed "$seed"
done

# -------------------------------------------------------------------- launch --
mkdir -p "$RUN_ROOT/logs"
status_file="$RUN_ROOT/launch_manifest.tsv"
printf 'seed\tcpu\tscreen_session\tlog\n' > "$status_file"

for index in "${!SEED_ARRAY[@]}"; do
    seed="${SEED_ARRAY[$index]}"
    core="${CORE_ARRAY[$index]}"
    session="${SESSION_PREFIX:-pspo_ll}_seed${seed}"
    log="$RUN_ROOT/logs/seed_$seed.log"
    metrics="$RUN_ROOT/lunarlander/seed_$seed/metrics.json"

    if [[ -f "$metrics" && "$SKIP_EXISTING" == "1" ]]; then
        echo "seed $seed: already complete ($metrics)"
        continue
    fi
    if screen -list | grep -q "\.${session}[[:space:]]"; then
        echo "seed $seed: screen session $session already running; skipping" >&2
        continue
    fi

    cmd=(
        taskset -c "$core" "$PYTHON"
        "$REPO_DIR/projects/safe_policy_optimisation/stages/train_pspo_continuous.py"
        --env-id LunarLander-v3
        --env-kwargs '{"continuous":false}'
        --continuous-shield lunarlander-descent
        --continuous-shield-config
        "{\"critical_height\": $CRITICAL_HEIGHT, \"safe_min_vertical_speed\": $SAFE_MIN_VERTICAL_SPEED}"
        --lunarlander-unsafe-height "$UNSAFE_HEIGHT"
        --lunarlander-unsafe-vertical-speed "$UNSAFE_VERTICAL_SPEED"
        --mountaincar-shaped-reward false
        --base-policy-path "$INIT_ROOT/seed_$seed/base_policy.pt"
        --certificate-dataset "$cert_path"
        --verify-first "$VERIFY_FIRST"
        --directional "$DIRECTIONAL"
        --region-mode "$REGION_MODE"
        --freq "$FREQ"
        --safe-region-shape "$SAFE_REGION_SHAPE"
        --segment-tolerance "$SEGMENT_TOLERANCE"
        --segment-splits "$SEGMENT_SPLITS"
        --segment-max-splits "$SEGMENT_MAX_SPLITS"
        --n-iters "$RASHOMON_N_ITERS"
        --rashomon-checkpoint 100
        --rashomon-batch-size auto
        --rashomon-multi-label-mode "$RASHOMON_MULTI_LABEL_MODE"
        --surrogate "$RASHOMON_SURROGATE"
        --growth-method "$GROWTH_METHOD"
        --certification-method "$CERTIFICATION_METHOD"
        --total-timesteps "$TOTAL_TIMESTEPS"
        --learning-rate "$LEARNING_RATE"
        --n-steps "$N_STEPS"
        --batch-size "$BATCH_SIZE"
        --n-epochs "$N_EPOCHS"
        --ent-coef "$ENT_COEF"
        --eval-episodes "$EVAL_EPISODES"
        --success-reward-threshold "$SUCCESS_REWARD_THRESHOLD"
        --evaluation-policy unshielded
        --early-stop-eval-policy unshielded
        --early-stop-eval-freq 25000
        --early-stop-eval-episodes 20
        --early-stop-success-rate 0.9
        --curve-eval-freq 25000
        --curve-eval-episodes 20
        --seed "$seed"
        --device cpu
        --output-dir "$RUN_ROOT/lunarlander"
        --run-id "seed_$seed"
    )

    # screen runs the command through a shell, so every argument has to survive
    # a second round of word splitting: the JSON arguments contain quotes,
    # braces, and spaces that would otherwise be mangled.
    printf -v cmd_str '%q ' "${cmd[@]}"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'dry-run seed %s core %s session %s:\n  %s\n' \
            "$seed" "$core" "$session" "$cmd_str"
        continue
    fi

    screen -dmS "$session" bash -c \
        "cd $(printf '%q' "$REPO_DIR") && $cmd_str > $(printf '%q' "$log") 2>&1; echo \"exit=\$?\" >> $(printf '%q' "$log")"
    printf '%s\t%s\t%s\t%s\n' "$seed" "$core" "$session" "$log" >> "$status_file"
    echo "seed $seed: launched in screen '$session' on core $core -> $log"
done

echo
echo "manifest: $status_file"
echo "attach:   screen -r ${SESSION_PREFIX:-pspo_ll}_seed0"
echo "list:     screen -ls | grep ${SESSION_PREFIX:-pspo_ll}_seed"
