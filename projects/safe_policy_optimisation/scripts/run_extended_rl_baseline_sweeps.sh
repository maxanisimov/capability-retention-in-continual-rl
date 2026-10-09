#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

OUT_BASE="${PAPER_OUT_BASE:-projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/rl_baselines_extended_budgets}"
SEED_LIST="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
METHOD_LIST="${METHOD_GROUPS:-ppo,baselines_lag,cpo,shielded}"
PARALLEL_JOBS="${SWEEP_PARALLEL:-40}"
FIRST_CPU="${CPU_OFFSET:-14}"
BRIDGE_TIMESTEPS="${BRIDGE_CROSSING_V2_TIMESTEPS:-1000000}"
MINIPACMAN_TIMESTEPS="${MINIPACMAN_TIMESTEPS:-2000000}"

cd "$REPO_ROOT"

export SEEDS="$SEED_LIST"
export METHOD_GROUPS="$METHOD_LIST"
export PAPER_OUT_BASE="$OUT_BASE"
export SWEEP_PARALLEL="$PARALLEL_JOBS"
export CPU_OFFSET="$FIRST_CPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_sweep() {
    local environment="$1"
    local timesteps="$2"

    export ENVS="$environment"
    export SMOKE_TIMESTEPS="$timesteps"

    "$REPO_ROOT/.venv/bin/python" \
        projects/safe_policy_optimisation/scripts/run_seed_experiments.py
}

run_sweep bridge_crossing_v2 "$BRIDGE_TIMESTEPS"
run_sweep mini_pacman "$MINIPACMAN_TIMESTEPS"
