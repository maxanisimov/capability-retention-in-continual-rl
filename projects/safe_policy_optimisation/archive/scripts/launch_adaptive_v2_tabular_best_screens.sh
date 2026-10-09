#!/usr/bin/env bash
set -euo pipefail

REPO=/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning
SCRIPT=projects/safe_policy_optimisation/scripts/run_adaptive_v2_tabular_best_one_env.sh
RUN_NAME="${RUN_NAME:-adaptive_v2_tabular_best_precomputed}"
SESSION_PREFIX="${SESSION_PREFIX:-pspo_v2}"
RASHOMON_TOTAL_ITERS="${RASHOMON_TOTAL_ITERS:-}"
RASHOMON_INITIAL_N_ITERS="${RASHOMON_INITIAL_N_ITERS:-}"
RASHOMON_RECOMPUTE_N_ITERS="${RASHOMON_RECOMPUTE_N_ITERS:-}"
CORE_OFFSET="${CORE_OFFSET:-0}"
LOGROOT=projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/${RUN_NAME}/_screen_logs

cd "$REPO"
mkdir -p "$LOGROOT"

launch() {
    local env_name="$1"
    local core_start="$2"
    local session="$3"
    local log_file="$4"

    if screen -ls | grep -F ".${session}" >/dev/null; then
        echo "Refusing to launch ${session}: a matching screen session already exists." >&2
        return 1
    fi

    screen -L -Logfile "$LOGROOT/$log_file" -dmS "$session" \
        bash -lc "cd '$REPO' && ENV_NAME='$env_name' CORE_START='$core_start' RUN_NAME='$RUN_NAME' RASHOMON_TOTAL_ITERS='$RASHOMON_TOTAL_ITERS' RASHOMON_INITIAL_N_ITERS='$RASHOMON_INITIAL_N_ITERS' RASHOMON_RECOMPUTE_N_ITERS='$RASHOMON_RECOMPUTE_N_ITERS' bash '$SCRIPT'"
    echo "launched ${session}: ENV_NAME=${env_name}, cores ${core_start}-$((core_start + 9)), log ${LOGROOT}/${log_file}"
}

launch media_streaming $((CORE_OFFSET + 0)) "${SESSION_PREFIX}_media_streaming" media_streaming.screen.log
launch colour_bomb $((CORE_OFFSET + 10)) "${SESSION_PREFIX}_colour_bomb" colour_bomb.screen.log
launch colour_bomb_v2 $((CORE_OFFSET + 20)) "${SESSION_PREFIX}_colour_bomb_v2" colour_bomb_v2.screen.log
launch bridge_crossing $((CORE_OFFSET + 30)) "${SESSION_PREFIX}_bridge_crossing" bridge_crossing.screen.log
launch bridge_crossing_v2 $((CORE_OFFSET + 40)) "${SESSION_PREFIX}_bridge_crossing_v2" bridge_crossing_v2.screen.log
launch mini_pacman $((CORE_OFFSET + 50)) "${SESSION_PREFIX}_mini_pacman" mini_pacman.screen.log

screen -ls
