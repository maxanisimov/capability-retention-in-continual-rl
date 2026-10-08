#!/usr/bin/env bash
# Policy-initialisation ablations (no_entropy, ce_only) across all six
# environments and seeds 0-9.
#
#   run_policy_initialisation_ablations.sh probe    find idle cores
#   run_policy_initialisation_ablations.sh plan     review the assignment
#   run_policy_initialisation_ablations.sh launch   run the wave
#   run_policy_initialisation_ablations.sh status   progress
#   run_policy_initialisation_ablations.sh analyse  paired-seed report
#
# Extra arguments are passed straight through, e.g.
#   ... launch --variants ce_only --envs colour_bomb --seeds 0,1
#
# This is a convenience wrapper only. It adds no experiment logic and changes
# no defaults: results are identical to calling lab_cluster/pspo_initialisation.py
# directly with the same three paths.
set -uo pipefail

# The dispatcher refuses a dirty source, so jobs run from the clean committed
# worktree. Results, however, belong to the main checkout -- the tool's default
# output root is derived from its own location and would otherwise write into
# the throwaway worktree.
MAIN_ROOT="${MAIN_ROOT:-/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning}"
SOURCE_ROOT="${SOURCE_ROOT:-$MAIN_ROOT/artifacts/launch_worktrees/pspo_policy_initialisation_20260828}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_ROOT/artifacts/ablation_studies/pspo_four_ablations}"
HOSTS_TSV="${HOSTS_TSV:-/tmp/pspo_policy_initialisation_hosts.tsv}"
PHASE="${PHASE:-train}"

PY="$SOURCE_ROOT/.venv/bin/python"
TOOL="$SOURCE_ROOT/projects/safe_policy_optimisation/scripts/lab_cluster/pspo_initialisation.py"
ANALYSE="$SOURCE_ROOT/projects/safe_policy_optimisation/scripts/analyse_pspo_four_ablations.py"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \?//'
  exit "${1:-1}"
}

[ $# -ge 1 ] || usage
COMMAND="$1"
shift

# ssh-add needs a passphrase, so never start an agent here -- tell the user how.
need_agent() {
  local sock="/tmp/$USER-pspo-policy-init-agent.sock"
  if [ -z "${SSH_AUTH_SOCK:-}" ] && [ -S "$sock" ]; then
    export SSH_AUTH_SOCK="$sock"
  fi
  if ! ssh-add -l >/dev/null 2>&1; then
    cat >&2 <<EOF
No usable ssh-agent. Start one and load your key, then re-run:

  ssh-agent -a $sock
  export SSH_AUTH_SOCK=$sock
  ssh-add ~/.ssh/id_ed25519
EOF
    exit 2
  fi
}

common=(--source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT")

case "$COMMAND" in
  probe)
    need_agent
    exec "$PY" "$TOOL" probe --source-root "$SOURCE_ROOT" --out "$HOSTS_TSV" \
      --minimum-idle 90 --reserve 2 --samples 5 "$@"
    ;;
  plan)
    need_agent
    exec "$PY" "$TOOL" dispatch --hosts-tsv "$HOSTS_TSV" "${common[@]}" \
      --phase "$PHASE" "$@"
    ;;
  launch)
    need_agent
    exec "$PY" "$TOOL" dispatch --hosts-tsv "$HOSTS_TSV" "${common[@]}" \
      --phase "$PHASE" --launch "$@"
    ;;
  status)
    need_agent
    exec "$PY" "$TOOL" status --output-root "$OUTPUT_ROOT" --check-hosts "$@"
    ;;
  analyse|analyze)
    exec "$PY" "$ANALYSE" --root "$OUTPUT_ROOT" \
      --variants control no_entropy ce_only \
      --output-dir "$OUTPUT_ROOT/analysis/policy_initialisation" "$@"
    ;;
  -h|--help|help)
    usage 0
    ;;
  *)
    echo "unknown command: $COMMAND" >&2
    usage
    ;;
esac
