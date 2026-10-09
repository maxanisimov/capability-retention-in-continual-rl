#!/usr/bin/env bash
set -euo pipefail

echo "warning: run_pspo_adaptive_one_env.sh is deprecated; use run_pspo_one_env.sh" >&2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_pspo_one_env.sh" "$@"
