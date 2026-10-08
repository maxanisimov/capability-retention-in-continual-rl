"""Spread a seed sweep across free lab machines so every job runs concurrently.

``run_seed_experiments.py`` already parallelises one (env, seed, method-group)
job per core, but it only ever uses the cores of the machine it was started on.
This dispatcher does the same job across many machines: it works out which
(seed, method-group) units are still missing, bin-packs them onto hosts with
spare cores, and launches one detached ``run_seed_experiments.py`` per host with
a disjoint ``CPU_OFFSET`` slice.

This works because ``/vol/bitbucket`` and ``/homes`` are NFS-mounted on every
DoC machine, so all hosts share one checkout, one virtualenv, and one output
tree.  Different hosts write to different ``seed*/<method>/`` directories, so
there is no write contention.

Typical use:

    scripts/lab_cluster/probe_hosts.sh --out /tmp/hosts.tsv
    python scripts/lab_cluster/dispatch.py --hosts /tmp/hosts.tsv    # plan only
    python scripts/lab_cluster/dispatch.py --hosts /tmp/hosts.tsv --launch

Re-running with ``--launch`` after a crash only redoes units whose
``metrics.json`` is still missing, so it doubles as the resume path.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# method group -> (metrics.json path relative to the seed dir, cores per seed)
#
# The core counts mirror METHOD_JOBS in run_seed_experiments.py: baselines_lag
# runs PPO-Lagrangian and PPO-PID-Lagrangian as two processes (--jobs 2), every
# other group is a single process.
GROUPS: dict[str, tuple[str, int]] = {
    "ppo": ("ppo_policy/metrics.json", 1),
    "baselines_lag": ("ppo_lagrangian/metrics.json", 2),
    "cpo": ("cpo/metrics.json", 1),
    "shielded": ("ppo_shield/metrics.json", 1),
}

DEFAULT_OUT_BASE = (
    "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/"
    "rl_baselines_extended_budgets"
)


@dataclass
class Unit:
    """One (seed, method-group) job, the smallest schedulable piece of work."""

    seed: int
    group: str

    @property
    def cores(self) -> int:
        return GROUPS[self.group][1]


@dataclass
class Host:
    name: str
    cores: int
    free_cores: int
    mem_avail_gb: float = 0.0
    mem_per_job_gb: float = 1.5
    units: list[Unit] = field(default_factory=list)

    @property
    def capacity(self) -> int:
        """Slots this host can take, limited by cores *and* by memory.

        Some lab machines are core-rich but RAM-poor (a 16-core box with 1.1 GB
        available), and filling their cores would OOM the jobs partway through
        a multi-hour run.
        """
        by_mem = int(self.mem_avail_gb / self.mem_per_job_gb)
        return max(0, min(self.free_cores, by_mem))

    @property
    def used(self) -> int:
        return sum(u.cores for u in self.units)

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    @property
    def cpu_base(self) -> int:
        """First core of our block: sit at the top of the range.

        Leaves the low-numbered cores to whoever else is on the machine, which
        matters on lab workstations someone may be sitting at.
        """
        return max(0, self.cores - self.free_cores)


def read_hosts(path: Path, min_free: int, mem_per_job_gb: float) -> list[Host]:
    """Read the TSV written by probe_hosts.sh, keeping only usable hosts."""
    hosts: list[Host] = []
    unusable = 0
    failed: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            # probe_hosts.sh puts DOWN/AUTH/ERROR in the cores column.
            status = row.get("cores", "")
            if status in {"DOWN", "AUTH", "ERROR", "UNREACHABLE"}:
                failed[status] = failed.get(status, 0) + 1
                continue
            if row.get("bitbucket") != "ok":
                unusable += 1
                continue
            host = Host(
                name=row["host"],
                cores=int(status),
                free_cores=int(row["free_cores"]),
                mem_avail_gb=float(row["mem_avail_gb"]),
                mem_per_job_gb=mem_per_job_gb,
            )
            if host.capacity < min_free:
                unusable += 1
                continue
            hosts.append(host)

    if failed:
        detail = ", ".join(f"{k.lower()}={v}" for k, v in sorted(failed.items()))
        print(f"probe recorded non-responding hosts ({detail})", file=sys.stderr)
    if failed.get("AUTH") and not hosts:
        print("every reachable host failed authentication -- fix credentials and "
              "re-probe before dispatching", file=sys.stderr)
    if unusable:
        print(f"skipping {unusable} host(s) with no /vol/bitbucket or too little "
              "core/memory capacity", file=sys.stderr)
    # Big hosts first: fewer machines involved means fewer things that can die.
    hosts.sort(key=lambda h: h.capacity, reverse=True)
    return hosts


def missing_units(out_base: Path, env: str, seeds: list[int], groups: list[str]) -> list[Unit]:
    """Units whose metrics.json does not exist yet."""
    units: list[Unit] = []
    for seed in seeds:
        seed_dir = out_base / env / f"seed{seed}"
        for group in groups:
            if not (seed_dir / GROUPS[group][0]).is_file():
                units.append(Unit(seed=seed, group=group))
    return units


def pack(units: list[Unit], hosts: list[Host]) -> list[Unit]:
    """First-fit-decreasing bin-pack. Returns the units that did not fit."""
    unplaced: list[Unit] = []
    # Wide units first, so a 2-core baselines_lag never gets stranded behind
    # a row of 1-core units that fragmented every host.
    for unit in sorted(units, key=lambda u: (-u.cores, u.seed, u.group)):
        for host in hosts:
            if host.remaining >= unit.cores:
                host.units.append(unit)
                break
        else:
            unplaced.append(unit)
    return unplaced


def build_launch_script(
    host: Host,
    env: str,
    out_base: str,
    timesteps: int,
    nice: int,
    dispatch_dir: str,
) -> str:
    """Generate the bash the host runs: one launcher per group, disjoint cores.

    Each ``run_seed_experiments.py`` invocation restarts its CPU cursor at 0, so
    concurrent invocations on one host must be handed non-overlapping
    CPU_OFFSETs or they all pin onto the same cores.
    """
    by_group: dict[str, list[int]] = {}
    for unit in host.units:
        by_group.setdefault(unit.group, []).append(unit.seed)

    lines = [
        "#!/usr/bin/env bash",
        f"# generated by dispatch.py for {host.name} at "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "set -uo pipefail",
        f"cd {shlex.quote(str(REPO))} || exit 1",
        "",
        "# Caches off the quota'd NFS home; the runner sets these per job too,",
        "# but the launcher process itself imports torch before forking.",
        'export XDG_CACHE_HOME="/vol/bitbucket/ma5923/.cache"',
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy",
        f"export PAPER_OUT_BASE={shlex.quote(out_base)}",
        f"export ENVS={shlex.quote(env)}",
        f"export SMOKE_TIMESTEPS={timesteps}",
        "",
        f"echo \"[$(date -Is)] {host.name}: starting, "
        f"{len(host.units)} units on cores {host.cpu_base}-{host.cpu_base + host.used - 1}\"",
        "",
    ]

    offset = host.cpu_base
    for group in GROUPS:  # stable order -> reproducible offsets
        seeds = sorted(by_group.get(group, []))
        if not seeds:
            continue
        width = GROUPS[group][1] * len(seeds)
        log = f"{dispatch_dir}/{host.name}__{group}.log"
        lines += [
            f"# {group}: {len(seeds)} seed(s) x {GROUPS[group][1]} core(s) "
            f"-> cores {offset}-{offset + width - 1}",
            f"SEEDS={shlex.quote(','.join(map(str, seeds)))} \\",
            f"  METHOD_GROUPS={shlex.quote(group)} \\",
            f"  CPU_OFFSET={offset} \\",
            f"  SWEEP_PARALLEL={len(seeds)} \\",
            f"  nice -n {nice} .venv/bin/python \\",
            "    projects/safe_policy_optimisation/scripts/run_seed_experiments.py \\",
            f"    > {shlex.quote(log)} 2>&1 &",
            "",
        ]
        offset += width

    lines += [
        "wait",
        f"echo \"[$(date -Is)] {host.name}: all groups finished\"",
        f"rm -f {shlex.quote(dispatch_dir)}/{host.name}.pid",
    ]
    return "\n".join(lines) + "\n"


def launch(host: Host, script_path: Path, dispatch_dir: Path, timeout: int) -> bool:
    """Start the host's script detached, so it survives our SSH disconnecting."""
    log = dispatch_dir / f"{host.name}.log"
    pid = dispatch_dir / f"{host.name}.pid"
    remote = (
        f"setsid nohup bash {shlex.quote(str(script_path))} "
        f"< /dev/null >> {shlex.quote(str(log))} 2>&1 & "
        f"echo $! > {shlex.quote(str(pid))}"
    )
    cmd = [
        "ssh", "-n",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "LogLevel=ERROR",
        f"{host.name}.doc.ic.ac.uk",
        f"bash -lc {shlex.quote(remote)}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {host.name}: TIMEOUT starting", file=sys.stderr)
        return False
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        print(f"  {host.name}: FAILED rc={proc.returncode} {err[:1]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hosts", type=Path, required=True,
                   help="TSV from probe_hosts.sh")
    p.add_argument("--env", default="mini_pacman", help="short env name")
    p.add_argument("--out-base", default=DEFAULT_OUT_BASE,
                   help="repo-relative output root")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--min-free", type=int, default=2,
                   help="ignore hosts with less usable capacity than this")
    p.add_argument("--mem-per-job-gb", type=float, default=1.5,
                   help="RAM budgeted per training process; caps how many slots "
                        "a RAM-poor host is given regardless of its free cores")
    p.add_argument("--max-hosts", type=int, default=0,
                   help="cap the number of machines used (0 = no cap)")
    p.add_argument("--nice", type=int, default=5,
                   help="niceness for training jobs; keeps lab machines usable")
    p.add_argument("--launch", action="store_true",
                   help="actually start the jobs (default: print the plan only)")
    p.add_argument("--ssh-timeout", type=int, default=30)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    groups = args.groups.split(",")
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        p.error(f"unknown groups {unknown}; valid: {sorted(GROUPS)}")

    out_base_abs = REPO / args.out_base
    units = missing_units(out_base_abs, args.env, seeds, groups)
    total = len(seeds) * len(groups)
    if not units:
        print(f"nothing to do: all {total} units already have metrics.json")
        return 0

    sys.stdout.flush()
    hosts = read_hosts(args.hosts, args.min_free, args.mem_per_job_gb)
    if args.max_hosts:
        hosts = hosts[: args.max_hosts]
    if not hosts:
        print("no usable hosts in the probe file", file=sys.stderr)
        return 1

    unplaced = pack(units, hosts)
    used = [h for h in hosts if h.units]

    need = sum(u.cores for u in units)
    have = sum(h.capacity for h in hosts)
    print(f"env={args.env} timesteps={args.timesteps:,}")
    print(f"units: {len(units)} missing of {total} "
          f"({need} slots needed, {have} usable across {len(hosts)} hosts "
          f"at {args.mem_per_job_gb} GB/job)")
    print()

    dispatch_dir = out_base_abs / "_dispatch"
    for host in used:
        by_group: dict[str, list[int]] = {}
        for unit in host.units:
            by_group.setdefault(unit.group, []).append(unit.seed)
        detail = "  ".join(
            f"{g}=[{','.join(map(str, sorted(s)))}]" for g, s in sorted(by_group.items())
        )
        print(f"{host.name:<10} {host.used:>2}/{host.capacity} slots "
              f"({host.cores}c, {host.mem_avail_gb:.0f}G, "
              f"block {host.cpu_base}-{host.cpu_base + host.used - 1})  {detail}")

    if unplaced:
        print()
        sys.stdout.flush()
        print(f"WARNING: {len(unplaced)} units do not fit and will NOT run:", file=sys.stderr)
        for unit in unplaced:
            print(f"  seed{unit.seed}/{unit.group}", file=sys.stderr)
        print("  probe more hosts, or lower RESERVE in probe_hosts.sh", file=sys.stderr)

    if not args.launch:
        print()
        print("plan only - re-run with --launch to start")
        return 0

    dispatch_dir.mkdir(parents=True, exist_ok=True)
    print()
    started = 0
    for host in used:
        script_path = dispatch_dir / f"{host.name}.sh"
        script_path.write_text(
            build_launch_script(host, args.env, args.out_base, args.timesteps,
                                args.nice, str(dispatch_dir)),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        if launch(host, script_path, dispatch_dir, args.ssh_timeout):
            started += 1
            print(f"  {host.name}: started ({len(host.units)} units)")

    manifest = {
        "dispatched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": args.env,
        "timesteps": args.timesteps,
        "hosts_started": started,
        "hosts_planned": len(used),
        "units_unplaced": [f"seed{u.seed}/{u.group}" for u in unplaced],
        "assignment": {
            h.name: [f"seed{u.seed}/{u.group}" for u in h.units] for h in used
        },
    }
    (dispatch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"started {started}/{len(used)} hosts; manifest in {dispatch_dir}/manifest.json")
    print("track with: python "
          "projects/safe_policy_optimisation/scripts/lab_cluster/status.py")
    return 0 if started == len(used) else 1


if __name__ == "__main__":
    raise SystemExit(main())
