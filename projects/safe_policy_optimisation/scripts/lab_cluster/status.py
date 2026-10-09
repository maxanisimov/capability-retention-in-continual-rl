"""Report progress of a sweep dispatched across lab machines.

Completion is read straight off the shared output tree (a unit is done when its
``metrics.json`` exists), so it works without touching any host.  Pass
``--check-hosts`` to additionally SSH each dispatched machine and confirm its
launcher is still alive - the failure mode worth catching is a host that died
hours ago while the others carried on.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispatch import DEFAULT_OUT_BASE, GROUPS  # noqa: E402


def unit_done(out_base: Path, env: str, seed: int, group: str) -> bool:
    return (out_base / env / f"seed{seed}" / GROUPS[group][0]).is_file()


def host_alive(host: str, timeout: int) -> str:
    """Return 'alive (N procs)', 'DEAD', or an error string."""
    # The bracket stops the pattern matching the wrapper shell that runs it,
    # which would otherwise add a phantom process and report a host whose jobs
    # have all died as "alive (1 procs)".
    remote = 'pgrep -c -u $USER -f "[r]un_experiment" || true'
    cmd = [
        "ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "LogLevel=ERROR", f"{host}.doc.ic.ac.uk",
        f"bash -lc {shlex.quote(remote)}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "ssh timeout"
    if proc.returncode != 0:
        return "ssh failed"
    count = (proc.stdout or "0").strip() or "0"
    return f"alive ({count} procs)" if count != "0" else "DEAD"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-base", default=DEFAULT_OUT_BASE)
    p.add_argument("--env", default="mini_pacman")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--check-hosts", action="store_true",
                   help="SSH each dispatched host to confirm its jobs are running")
    p.add_argument("--ssh-timeout", type=int, default=20)
    args = p.parse_args()

    out_base = REPO / args.out_base
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = list(GROUPS)

    print(f"{args.env} @ {args.out_base}")
    print()
    header = "seed   " + "".join(f"{g:<16}" for g in groups)
    print(header)
    print("-" * len(header))
    done_total = 0
    for seed in seeds:
        cells = []
        for group in groups:
            ok = unit_done(out_base, args.env, seed, group)
            done_total += int(ok)
            cells.append(f"{'done' if ok else '-':<16}")
        print(f"{seed:<7}" + "".join(cells))
    total = len(seeds) * len(groups)
    print()
    print(f"complete: {done_total}/{total} units "
          f"({100 * done_total / total:.0f}%)")

    manifest_path = out_base / "_dispatch" / "manifest.json"
    if not manifest_path.is_file():
        print("\n(no _dispatch/manifest.json - nothing dispatched from here yet)")
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"\ndispatched {manifest['dispatched_utc']} across "
          f"{manifest['hosts_started']} host(s)")

    if not args.check_hosts:
        print("(pass --check-hosts to verify the launchers are still running)")
        return 0

    hosts = sorted(manifest["assignment"])
    print()
    with ThreadPoolExecutor(max_workers=min(32, len(hosts))) as ex:
        states = list(ex.map(lambda h: host_alive(h, args.ssh_timeout), hosts))
    for host, state in zip(hosts, states):
        assigned = manifest["assignment"][host]
        outstanding = [
            u for u in assigned
            if not unit_done(out_base, args.env,
                             int(u.split("/")[0].removeprefix("seed")), u.split("/")[1])
        ]
        flag = "  <-- investigate" if state == "DEAD" and outstanding else ""
        print(f"{host:<10} {state:<18} "
              f"{len(assigned) - len(outstanding)}/{len(assigned)} units done{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
