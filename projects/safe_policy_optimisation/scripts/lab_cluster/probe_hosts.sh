#!/usr/bin/env bash
# Probe DoC lab machines for spare CPU capacity.
#
# Writes a TSV to stdout (and to --out if given), one row per host:
#
#   host  cores  load15  free_cores  mem_avail_gb  users  bitbucket
#
# ``free_cores`` is cores minus the 15-minute load average, minus the reserve
# kept for whoever else is on the box.  ``bitbucket`` is ok/MISSING; a host
# without /vol/bitbucket cannot run the sweep at all.
#
# Hosts that do not answer get ``cores`` set to a reason, never to a number:
#   DOWN  - TCP/22 unreachable (powered off, or off the network)
#   AUTH  - reachable but SSH rejected us (no agent / no Kerberos ticket)
#   ERROR - reachable and authenticated, but the probe command failed
#
# The distinction matters: a wall of AUTH means your credentials are not
# reaching ssh, and says nothing about how busy the lab is.
#
# Probing is read-only: it runs nproc/uptime/free/who and nothing else.

set -uo pipefail

DOMAIN="${DOMAIN:-doc.ic.ac.uk}"
RESERVE="${RESERVE:-2}"          # cores left free per host for other users
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-6}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-20}"
FANOUT="${FANOUT:-60}"
OUT=""

# Families of DoC lab machines. Override with HOSTS="a01 a02 ..." or --hosts.
DEFAULT_FAMILIES="${FAMILIES:-ash:40 oak:38 willow:20 beech:20 vertex:22 curve:10}"

usage() {
    cat <<'EOF'
usage: probe_hosts.sh [--out FILE] [--reserve N] [--hosts "h1 h2 ..."]

env overrides:
  FAMILIES  "name:count ..."  host families to expand (default ash/oak/willow/
                              beech/vertex/curve)
  HOSTS     explicit space-separated short host names (skips family expansion)
  RESERVE   cores to leave free per host (default 2)
  FANOUT    max concurrent ssh probes (default 60)
  SKIP_PREFLIGHT=1  probe even if no ssh-agent/Kerberos credentials are found

SSH must work non-interactively. Start an agent and export its socket into the
shell that runs this script -- exporting matters, the probes are subprocesses:

  ssh-agent -a /tmp/$USER-agent.sock
  export SSH_AUTH_SOCK=/tmp/$USER-agent.sock
  ssh-add ~/.ssh/id_ed25519
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --reserve) RESERVE="$2"; shift 2 ;;
        --hosts) HOSTS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Preflight: catch missing credentials here rather than as 150 identical
# failures that look like a busy lab.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    have_creds=0
    if [ -n "${SSH_AUTH_SOCK:-}" ] && ssh-add -l >/dev/null 2>&1; then
        have_creds=1
    elif klist -s 2>/dev/null; then
        have_creds=1
    fi
    if [ "$have_creds" -eq 0 ]; then
        echo "error: no usable SSH credentials in this shell." >&2
        echo "" >&2
        if [ -z "${SSH_AUTH_SOCK:-}" ]; then
            echo "  SSH_AUTH_SOCK is unset. If you already started an agent," >&2
            echo "  you still need to export it here:" >&2
            echo "    export SSH_AUTH_SOCK=/tmp/$USER-agent.sock" >&2
        else
            echo "  SSH_AUTH_SOCK=$SSH_AUTH_SOCK has no keys loaded. Add one:" >&2
            echo "    ssh-add ~/.ssh/id_ed25519" >&2
        fi
        echo "" >&2
        echo "  (or run kinit for Kerberos, or SKIP_PREFLIGHT=1 to probe anyway)" >&2
        exit 3
    fi
fi

build_host_list() {
    if [ -n "${HOSTS:-}" ]; then
        printf '%s\n' $HOSTS
        return
    fi
    local spec name count i
    for spec in $DEFAULT_FAMILIES; do
        name="${spec%%:*}"
        count="${spec##*:}"
        for i in $(seq -w 1 "$count"); do
            printf '%s%s\n' "$name" "$i"
        done
    done
}

# One probe. Prints a TSV row; failures carry a reason in the cores column.
probe_one() {
    local short="$1" fqdn="$1.$DOMAIN" out err rc

    # Cheap TCP check first: a powered-off host otherwise burns the full ssh
    # timeout and is indistinguishable from a slow one.
    if ! timeout "$CONNECT_TIMEOUT" bash -c "exec 3<>/dev/tcp/$fqdn/22" 2>/dev/null; then
        printf '%s\tDOWN\t-\t-\t-\t-\t-\n' "$short"
        return
    fi

    err=$(mktemp "${TMPDIR:-/tmp}/probe_err.XXXXXX")
    out=$(timeout "$PROBE_TIMEOUT" ssh -n \
              -o BatchMode=yes \
              -o StrictHostKeyChecking=accept-new \
              -o ConnectTimeout="$CONNECT_TIMEOUT" \
              -o LogLevel=ERROR \
              "$fqdn" \
              'printf "%s|%s|%s|%s\n" \
                 "$(nproc)" \
                 "$(cut -d" " -f3 /proc/loadavg)" \
                 "$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)" \
                 "$(who | awk "{print \$1}" | sort -u | wc -l)"; \
               [ -d /vol/bitbucket/ma5923 ] && echo ok || echo MISSING' 2>"$err")
    rc=$?

    if [ $rc -ne 0 ] || [ -z "$out" ]; then
        local reason="ERROR"
        grep -qi 'permission denied\|no supported authentication' "$err" && reason="AUTH"
        grep -qi 'timed out\|no route to host\|refused' "$err" && reason="DOWN"
        rm -f "$err"
        printf '%s\t%s\t-\t-\t-\t-\t-\n' "$short" "$reason"
        return
    fi
    rm -f "$err"

    local stats bb cores load15 mem users free
    stats=$(printf '%s\n' "$out" | sed -n 1p)
    bb=$(printf '%s\n' "$out" | sed -n 2p)
    cores=${stats%%|*}; stats=${stats#*|}
    load15=${stats%%|*}; stats=${stats#*|}
    mem=${stats%%|*}
    users=${stats##*|}

    free=$(awk -v c="$cores" -v l="$load15" -v r="$RESERVE" \
               'BEGIN { f = int(c - l - r); if (f < 0) f = 0; print f }')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$short" "$cores" "$load15" "$free" "$mem" "$users" "$bb"
}

export -f probe_one
export DOMAIN RESERVE CONNECT_TIMEOUT PROBE_TIMEOUT

tmp=$(mktemp "${TMPDIR:-/tmp}/probe_hosts.XXXXXX")
trap 'rm -f "$tmp"' EXIT

echo "probing $(build_host_list | wc -l) hosts (fanout=$FANOUT, reserve=$RESERVE)..." >&2
build_host_list | xargs -P "$FANOUT" -I{} bash -c 'probe_one "$@"' _ {} > "$tmp"

is_bad() { printf '%s\n' "DOWN AUTH ERROR"; }

{
    printf 'host\tcores\tload15\tfree_cores\tmem_avail_gb\tusers\tbitbucket\n'
    # Usable hosts first, sorted by spare capacity; failures last.
    grep -Ev $'\t(DOWN|AUTH|ERROR)\t' "$tmp" | sort -t$'\t' -k4,4nr
    grep -E $'\t(DOWN|AUTH|ERROR)\t' "$tmp" | sort
} | { [ -n "$OUT" ] && tee "$OUT" || cat; }

n_ok=$(grep -Ev $'\t(DOWN|AUTH|ERROR)\t' "$tmp" | wc -l)
n_down=$(grep -cE $'\tDOWN\t' "$tmp")
n_auth=$(grep -cE $'\tAUTH\t' "$tmp")
n_err=$(grep -cE $'\tERROR\t' "$tmp")
usable=$(grep -Ev $'\t(DOWN|AUTH|ERROR)\t' "$tmp" \
         | awk -F'\t' '$7 == "ok" { s += $4 } END { print s + 0 }')

{
    echo ""
    echo "responded: $n_ok    down: $n_down    auth-failed: $n_auth    error: $n_err"
    if [ "$n_auth" -gt 0 ] && [ "$n_ok" -eq 0 ]; then
        echo ""
        echo "ALL reachable hosts rejected authentication -- this is a credentials"
        echo "problem, not a capacity result. Export SSH_AUTH_SOCK into this shell"
        echo "and load your key, then re-probe."
    else
        echo "total free cores on responding hosts with /vol/bitbucket: $usable"
    fi
    [ -n "$OUT" ] && echo "wrote $OUT"
} >&2
exit 0
