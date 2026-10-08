"""Probe, dispatch, and monitor the PSPO policy-initialisation ablation.

The production experiment has two dependent waves: twelve shared base-policy
initialisers, followed by 120 independently pinned PPO runs.  Capacity is
measured per logical CPU with ``mpstat``; load averages are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPT_REPO / "core"))
sys.path.insert(0, str(SCRIPT_REPO))
from projects.safe_policy_optimisation.scripts import run_pspo_four_ablations as suite

REPO = SCRIPT_REPO
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/ablation_studies/pspo_four_ablations"
DEFAULT_FAMILIES = "ash:40 oak:38 willow:20 beech:20 vertex:22 curve:10"
ENVIRONMENTS = suite.ENVIRONMENTS
VARIANTS = ("no_entropy", "ce_only")
ARCHITECTURE = "two_hidden"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
LAUNCH_ENV_KEYS = (
    "ENV_NAME",
    "SEEDS",
    "CPU_IDS",
    "ARCHITECTURE",
    "RUN_NAME",
    "OUTPUT_BASE",
    "ADAPTIVE_FREQ",
    "TOTAL_TIMESTEPS",
    "SKIP_EXISTING",
    "DRY_RUN",
    "PSPO_SMOKE",
    "PREPARE_BASE_ONLY",
    "BASE_POLICY_PATH",
    "BC_MARGIN_LOSS_WEIGHT",
    "BC_SAFE_ACTION_ENTROPY_WEIGHT",
    "RASHOMON_MULTI_LABEL_MODE",
    "REGION_REFRESH",
    "DIRECTIONAL_RASHOMON_GROWTH",
    "STOP_WHEN_PROPOSAL_CONTAINED",
    "VERIFY_FIRST",
    "AUDIT_CANDIDATES_EXACTLY",
    "RASHOMON_N_ITERS",
    *THREAD_ENV,
)
SOURCE_FILES = (
    "core/provably_safe_policy_optimisation/adaptive_safe_ppo.py",
    "core/provably_safe_policy_optimisation/adaptive_safe_ppo_v2.py",
    "projects/safe_policy_optimisation/stages/compute_shield_rashomon_set.py",
    "projects/safe_policy_optimisation/stages/train_pspo.py",
    "projects/safe_policy_optimisation/utils/learning_curves.py",
    "projects/safe_policy_optimisation/utils/pspo_defaults.py",
    "projects/safe_policy_optimisation/utils/pspo_launcher.py",
    "projects/safe_policy_optimisation/docs/pspo_precomputed/pspo_precomputed_best_hyperparameters.json",
    "projects/safe_policy_optimisation/docs/pspo_precomputed/best_pspo_precomputed_by_architecture_current.json",
    "projects/safe_policy_optimisation/scripts/run_pspo_one_env.sh",
    "projects/safe_policy_optimisation/scripts/run_pspo_four_ablations.py",
    "projects/safe_policy_optimisation/scripts/analyse_pspo_four_ablations.py",
    "projects/safe_policy_optimisation/scripts/lab_cluster/pspo_initialisation.py",
    "projects/safe_policy_optimisation/docs/pspo_policy_initialisation_launch.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_mpstat_idle(output: str) -> dict[int, float]:
    idle: dict[int, float] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "Average:" and fields[1].isdigit():
            try:
                idle[int(fields[1])] = float(fields[-1])
            except ValueError:
                pass
    if not idle:
        raise ValueError("mpstat output has no average per-CPU samples")
    return idle


def expand_hosts(explicit: str | None, families: str) -> list[str]:
    if explicit:
        result = explicit.replace(",", " ").split()
    else:
        result = []
        for item in families.split():
            family, raw_count = item.rsplit(":", 1)
            result.extend(f"{family}{index:02d}" for index in range(1, int(raw_count) + 1))
    if len(result) != len(set(result)):
        raise ValueError("host list contains duplicates")
    return result


def ssh_command(host: str, remote: str, *, domain: str) -> list[str]:
    return [
        "ssh", "-n", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR",
        f"{host}.{domain}", f"bash -lc {shlex.quote(remote)}",
    ]


def check_credentials() -> None:
    agent = bool(os.environ.get("SSH_AUTH_SOCK")) and subprocess.run(
        ["ssh-add", "-l"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    kerberos = bool(shutil.which("klist")) and subprocess.run(
        ["klist", "-s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    if not agent and not kerberos:
        raise RuntimeError(
            "no non-interactive SSH credential; export SSH_AUTH_SOCK and run "
            "ssh-add ~/.ssh/id_ed25519 before probing or launching"
        )


@dataclass(frozen=True)
class ProbeResult:
    host: str
    status: str
    cores: int = 0
    load15: float = 0.0
    mem_avail_gb: float = 0.0
    users: int = 0
    bitbucket: str = "-"
    source: str = "-"
    venv: str = "-"
    idle_by_cpu: dict[int, float] = field(default_factory=dict)
    error: str = ""

    def eligible_cpus(self, *, minimum_idle: float, reserve: int) -> list[int]:
        return sorted(
            cpu for cpu, idle in self.idle_by_cpu.items()
            if cpu >= reserve and idle >= minimum_idle
        )


def probe_host(
    host: str,
    *,
    domain: str,
    source_root: Path,
    samples: int,
    timeout: int,
) -> ProbeResult:
    quoted_source = shlex.quote(str(source_root))
    remote = (
        "set -e; command -v mpstat >/dev/null; "
        "printf '__PSPO_META__\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
        '"$(nproc)" "$(cut -d\" \" -f3 /proc/loadavg)" '
        '"$(awk \'/MemAvailable/{printf "%.3f", $2/1048576}\' /proc/meminfo)" '
        '"$(who | awk \'{print $1}\' | sort -u | wc -l)" '
        '"$([ -d /vol/bitbucket/ma5923 ] && echo ok || echo MISSING)" '
        f'"$([ -d {quoted_source} ] && echo ok || echo MISSING)" '
        f'"$([ -x {quoted_source}/.venv/bin/python ] && echo ok || echo MISSING)"; '
        f"LC_ALL=C mpstat -P ALL 1 {int(samples)}"
    )
    try:
        completed = subprocess.run(
            ssh_command(host, remote, domain=domain), capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(host=host, status="DOWN", error="ssh timeout")
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        lowered = message.lower()
        status = "AUTH" if "permission denied" in lowered else "ERROR"
        return ProbeResult(host=host, status=status, error=message[:300])
    metadata = next(
        (line for line in completed.stdout.splitlines() if line.startswith("__PSPO_META__\t")),
        None,
    )
    if metadata is None:
        return ProbeResult(host=host, status="ERROR", error="missing probe metadata")
    fields = metadata.split("\t")
    try:
        idle = parse_mpstat_idle(completed.stdout)
        return ProbeResult(
            host=host, status="ok", cores=int(fields[1]), load15=float(fields[2]),
            mem_avail_gb=float(fields[3]), users=int(fields[4]), bitbucket=fields[5],
            source=fields[6], venv=fields[7], idle_by_cpu=idle,
        )
    except (IndexError, ValueError) as exc:
        return ProbeResult(host=host, status="ERROR", error=str(exc))


def write_probe_tsv(
    path: Path,
    results: Iterable[ProbeResult],
    *,
    minimum_idle: float,
    reserve: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "host", "cores", "load15", "free_cores", "mem_avail_gb", "users",
            "bitbucket", "idle_cpu_ids", "probe_minimum_idle", "reserved_low_cpus",
            "source", "venv", "status", "error",
        ))
        for result in sorted(results, key=lambda item: item.host):
            if result.status != "ok":
                writer.writerow((
                    result.host, result.status, "-", 0, "-", "-", "-", "-",
                    minimum_idle, reserve, "-", "-", result.status, result.error,
                ))
                continue
            cpus = result.eligible_cpus(minimum_idle=minimum_idle, reserve=reserve)
            writer.writerow((
                result.host, result.cores, f"{result.load15:.2f}", len(cpus),
                f"{result.mem_avail_gb:.3f}", result.users, result.bitbucket,
                ",".join(map(str, cpus)), minimum_idle, reserve,
                result.source, result.venv, "ok", "",
            ))


@dataclass
class Host:
    name: str
    cpus: list[int]
    mem_avail_gb: float
    mem_per_job_gb: float
    assignments: list["Assignment"] = field(default_factory=list)

    @property
    def capacity(self) -> int:
        return min(len(self.cpus), int(self.mem_avail_gb / self.mem_per_job_gb))

    @property
    def remaining(self) -> int:
        return self.capacity - len(self.assignments)


@dataclass(frozen=True)
class Unit:
    variant: str
    environment: str
    seed: int | None
    phase: str

    @property
    def key(self) -> str:
        suffix = "base" if self.seed is None else f"seed{self.seed}"
        return f"{self.variant}/{self.environment}/{suffix}"


@dataclass(frozen=True)
class Assignment:
    unit: Unit
    cpu: int


def read_hosts(
    path: Path, *, mem_per_job_gb: float, minimum_idle: float, reserve: int,
) -> list[Host]:
    hosts: list[Host] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "ok":
                continue
            if any(row.get(key) != "ok" for key in ("bitbucket", "source", "venv")):
                continue
            try:
                recorded_idle = float(row["probe_minimum_idle"])
                recorded_reserve = int(row["reserved_low_cpus"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("host TSV lacks exact probe-policy provenance") from exc
            if recorded_idle != minimum_idle or recorded_reserve != reserve:
                raise ValueError(
                    "host TSV probe policy differs from dispatch policy: "
                    f"recorded idle={recorded_idle}, reserve={recorded_reserve}; "
                    f"requested idle={minimum_idle}, reserve={reserve}"
                )
            cpus = [int(value) for value in row.get("idle_cpu_ids", "").split(",") if value]
            host = Host(row["host"], cpus, float(row["mem_avail_gb"]), mem_per_job_gb)
            if host.capacity:
                hosts.append(host)
    hosts.sort(key=lambda item: (-item.capacity, item.name))
    return hosts


def spread(units: list[Unit], hosts: list[Host]) -> list[Unit]:
    """Allocate at most one job per host per pass, using exact CPU IDs."""

    cursor = 0
    while cursor < len(units):
        placed_this_pass = 0
        for host in hosts:
            if cursor >= len(units):
                break
            if host.remaining <= 0:
                continue
            cpu = host.cpus[len(host.assignments)]
            host.assignments.append(Assignment(units[cursor], cpu))
            cursor += 1
            placed_this_pass += 1
        if placed_this_pass == 0:
            break
    return units[cursor:]


def base_dir(output_root: Path, variant: str, environment: str) -> Path:
    return output_root / variant / ARCHITECTURE / environment / "initial_base_policy"


def base_failure_marker(output_root: Path, variant: str, environment: str) -> Path:
    return base_dir(output_root, variant, environment) / "INITIALISER_FAILED"


def validate_base(output_root: Path, variant: str, environment: str) -> dict[str, Any]:
    directory = base_dir(output_root, variant, environment)
    policy = directory / "base_policy.pt"
    summary_path = directory / "summary.json"
    dataset_path = directory / "safe_behaviour_dataset.pt"
    if base_failure_marker(output_root, variant, environment).exists():
        raise ValueError(f"initializer previously failed: {directory}")
    if not policy.is_file() or not summary_path.is_file():
        raise FileNotFoundError(directory)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_margin = 1.0 if variant == "no_entropy" else 0.0
    # ce_only is graded in "any" mode: cross-entropy optimises the best safe
    # logit, so the worst-safe-logit ("all") criterion tests something its loss
    # never touches. The Rashomon set uses the same mode for consistency.
    expected_stopping = (
        "target_margin" if variant == "no_entropy" else "any_safe_feasibility"
    )
    expected_margin_mode = "all" if variant == "no_entropy" else "any"
    base = summary.get("base_policy", {})
    architecture = summary.get("architecture", {})
    dataset = summary.get("dataset", {})
    shield = Path(str(summary.get("shield_path", "")))
    checks = {
        "base_policy_only": summary.get("base_policy_only") is True,
        "dataset_artifact": dataset_path.is_file(),
        "shield_hash": shield.is_file() and summary.get("shield_sha256") == sha256(shield),
        "two_hidden": int(architecture.get("n_hidden", -1)) == 2,
        "hidden_dim": int(architecture.get("hidden_dim", -1)) == 64,
        "one_hot_state": architecture.get("state_representation") == "one_hot_discrete_observation",
        "positive_dataset": int(dataset.get("dataset_size", 0)) > 0,
        "margin_mode": base.get("bc_margin_mode") == expected_margin_mode,
        "margin_objective": base.get("initialisation_objective", "margin") == "margin",
        "margin_loss_weight": float(base.get("margin_loss_weight", -1.0)) == expected_margin,
        "entropy_weight": float(base.get("safe_action_entropy_weight", -1.0)) == 0.0,
        "stopping_criterion": base.get("stopping_criterion") == expected_stopping,
        "reached_target": base.get("reached_target") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid {variant}/{environment} base metadata: {failed}")
    return {
        "path": str(policy.resolve()),
        "sha256": sha256(policy),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "checks": checks,
    }


def valid_metrics(path: Path) -> bool:
    try:
        return path.is_file() and isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, ValueError):
        return False


def outstanding_units(
    *, phase: str, output_root: Path, variants: list[str], environments: list[str],
    seeds: list[int],
) -> tuple[list[Unit], dict[str, dict[str, Any]]]:
    bases: dict[str, dict[str, Any]] = {}
    units: list[Unit] = []
    for variant in variants:
        for environment in environments:
            try:
                bases[f"{variant}/{environment}"] = validate_base(output_root, variant, environment)
                ready = True
            except FileNotFoundError:
                ready = False
            if phase == "base":
                if not ready:
                    units.append(Unit(variant, environment, None, phase))
                continue
            if not ready:
                raise ValueError(
                    f"training wave is blocked: base policy is missing for {variant}/{environment}"
                )
            for seed in seeds:
                metrics = output_root / variant / ARCHITECTURE / environment / f"seed{seed}" / "metrics.json"
                if not valid_metrics(metrics):
                    units.append(Unit(variant, environment, seed, phase))
    return units, bases


def git_state(source_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def source_hashes(source_root: Path) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"launch source is incomplete: {missing}")
    return {name: sha256(source_root / name) for name in SOURCE_FILES}


def validate_controls(environments: list[str], seeds: list[int]) -> dict[str, Any]:
    return {
        environment: suite.validate_control(
            environment, architecture=ARCHITECTURE, seeds=seeds
        )
        for environment in environments
    }


def command_environment(
    assignment: Assignment, *, output_root: Path, smoke: bool,
) -> dict[str, str]:
    unit = assignment.unit
    job = suite.variant_environment(
        unit.variant, unit.environment, unit.seed or 0, cpu=assignment.cpu,
        architecture=ARCHITECTURE, source=None, output_root=output_root,
        dry_run=False, total_timesteps_override=None, smoke=smoke,
    )
    env = {key: job.env[key] for key in LAUNCH_ENV_KEYS if key in job.env}
    env.update(THREAD_ENV)
    if unit.phase == "base":
        env["PREPARE_BASE_ONLY"] = "1"
        env["SEEDS"] = "0"
    else:
        env["PREPARE_BASE_ONLY"] = "0"
        env["BASE_POLICY_PATH"] = str(
            base_dir(output_root, unit.variant, unit.environment) / "base_policy.pt"
        )
    return env


def build_host_script(
    host: Host, *, source_root: Path, output_root: Path, dispatch_dir: Path,
    nice: int, smoke: bool,
) -> str:
    launcher = source_root / "projects/safe_policy_optimisation/scripts/run_pspo_one_env.sh"
    lines = [
        "#!/usr/bin/env bash", "set -uo pipefail", f"cd {shlex.quote(str(source_root))} || exit 1",
        "pids=()", "names=()", "",
    ]
    for assignment in host.assignments:
        unit = assignment.unit
        env = command_environment(assignment, output_root=output_root, smoke=smoke)
        env_args = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
        safe_name = unit.key.replace("/", "__")
        log = dispatch_dir / f"{host.name}__{safe_name}.log"
        marker = base_failure_marker(output_root, unit.variant, unit.environment) if unit.phase == "base" else None
        command = (
            f"env {env_args} nice -n {int(nice)} bash {shlex.quote(str(launcher))} "
            f"> {shlex.quote(str(log))} 2>&1"
        )
        if marker is not None:
            command = (
                f"( {command}; rc=$?; if [ $rc -ne 0 ]; then mkdir -p {shlex.quote(str(marker.parent))}; "
                f"printf '%s\\n' \"$(date -Is) rc=$rc\" > {shlex.quote(str(marker))}; fi; exit $rc )"
            )
        lines.extend((f"{command} &", "pids+=(\"$!\")", f"names+=({shlex.quote(unit.key)})", ""))
    lines.extend((
        "failed=0", 'for i in "${!pids[@]}"; do',
        '  if ! wait "${pids[$i]}"; then echo "FAILED ${names[$i]}"; failed=1; fi',
        "done", f"printf '%s\\n' \"$(date -Is) rc=$failed\" > {shlex.quote(str(dispatch_dir / (host.name + '.done')))}",
        f"rm -f {shlex.quote(str(dispatch_dir / (host.name + '.pid')))}", "exit $failed", "",
    ))
    return "\n".join(lines)


def revalidate(
    hosts: list[Host], *, domain: str, source_root: Path, minimum_idle: float,
    reserve: int, samples: int, timeout: int,
) -> list[str]:
    used = [host for host in hosts if host.assignments]
    with ThreadPoolExecutor(max_workers=min(32, len(used))) as executor:
        futures = [
            executor.submit(
                probe_host, host.name, domain=domain, source_root=source_root,
                samples=samples, timeout=timeout,
            )
            for host in used
        ]
        fresh = [future.result() for future in futures]
    errors: list[str] = []
    for planned, measured in zip(used, fresh):
        eligible = set(measured.eligible_cpus(minimum_idle=minimum_idle, reserve=reserve))
        assigned = {assignment.cpu for assignment in planned.assignments}
        if measured.status != "ok":
            errors.append(f"{planned.name}: {measured.status} {measured.error}")
        elif not assigned <= eligible:
            errors.append(f"{planned.name}: assigned CPUs became busy: {sorted(assigned - eligible)}")
        elif measured.source != "ok" or measured.venv != "ok":
            errors.append(f"{planned.name}: source={measured.source} venv={measured.venv}")
    return errors


def launch_host(host: Host, *, dispatch_dir: Path, domain: str, timeout: int) -> bool:
    script = dispatch_dir / f"{host.name}.sh"
    log = dispatch_dir / f"{host.name}.launcher.log"
    pid = dispatch_dir / f"{host.name}.pid"
    remote = (
        f"setsid nohup bash {shlex.quote(str(script))} < /dev/null "
        f">> {shlex.quote(str(log))} 2>&1 & echo $! > {shlex.quote(str(pid))}"
    )
    try:
        completed = subprocess.run(
            ssh_command(host.name, remote, domain=domain), capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def probe_main(args: argparse.Namespace) -> int:
    if not args.skip_auth_preflight:
        check_credentials()
    names = expand_hosts(args.hosts, args.families)
    with ThreadPoolExecutor(max_workers=min(args.fanout, len(names))) as executor:
        futures = [
            executor.submit(
                probe_host, name, domain=args.domain, source_root=args.source_root.resolve(),
                samples=args.samples, timeout=args.timeout,
            )
            for name in names
        ]
        results = [future.result() for future in futures]
    write_probe_tsv(
        args.out, results, minimum_idle=args.minimum_idle, reserve=args.reserve,
    )
    usable = [
        result for result in results
        if result.status == "ok" and result.bitbucket == result.source == result.venv == "ok"
    ]
    slots = sum(len(item.eligible_cpus(minimum_idle=args.minimum_idle, reserve=args.reserve)) for item in usable)
    print(f"probed {len(results)} hosts: {len(usable)} usable, {slots} exact idle CPU slots")
    print(f"wrote {args.out}")
    return 0


def dispatch_main(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    state = git_state(source_root)
    if state["dirty"] and not args.allow_dirty:
        raise RuntimeError("launch source is dirty; use a clean committed worktree")
    hashes = source_hashes(source_root)
    controls = validate_controls(args.envs, args.seeds)
    units, bases = outstanding_units(
        phase=args.phase, output_root=output_root, variants=args.variants,
        environments=args.envs, seeds=args.seeds,
    )
    total = len(args.variants) * len(args.envs) * (1 if args.phase == "base" else len(args.seeds))
    if not units:
        print(f"nothing to do: all {total} {args.phase} units are valid")
        return 0
    hosts = read_hosts(
        args.hosts_tsv, mem_per_job_gb=args.mem_per_job_gb,
        minimum_idle=args.minimum_idle, reserve=args.reserve,
    )
    if args.max_hosts:
        hosts = hosts[: args.max_hosts]
    unplaced = spread(units, hosts)
    used = [host for host in hosts if host.assignments]
    print(f"phase={args.phase}: {len(units)} outstanding of {total}; {sum(h.capacity for h in hosts)} slots")
    for host in used:
        details = ", ".join(f"cpu{item.cpu}:{item.unit.key}" for item in host.assignments)
        print(f"{host.name:<10} {details}")
    if unplaced:
        print(f"unplaced ({len(unplaced)}): " + ", ".join(unit.key for unit in unplaced), file=sys.stderr)
        if args.launch:
            raise RuntimeError("all outstanding units must fit; nothing was launched")
    if not args.launch:
        print("plan only; pass --launch after reviewing the exact host/CPU assignment")
        return 0
    check_credentials()
    stale = revalidate(
        used, domain=args.domain, source_root=source_root,
        minimum_idle=args.minimum_idle, reserve=args.reserve,
        samples=args.revalidate_samples, timeout=args.ssh_timeout,
    )
    if stale:
        raise RuntimeError("launch plan became stale; re-probe:\n" + "\n".join(stale))
    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dispatch_dir = output_root / "_policy_initialisation_dispatch" / f"{args.phase}_{launch_id}"
    dispatch_dir.mkdir(parents=True, exist_ok=False)
    for host in used:
        script = dispatch_dir / f"{host.name}.sh"
        script.write_text(build_host_script(
            host, source_root=source_root, output_root=output_root,
            dispatch_dir=dispatch_dir, nice=args.nice, smoke=args.smoke,
        ), encoding="utf-8")
        script.chmod(0o755)
    manifest = {
        "schema_version": 1, "created_at": utc_now(), "launch_id": launch_id,
        "phase": args.phase, "source_root": str(source_root), "output_root": str(output_root),
        "repository": state, "source_hashes": hashes,
        "probe_tsv": str(args.hosts_tsv.resolve()), "probe_sha256": sha256(args.hosts_tsv),
        "minimum_idle_percent": args.minimum_idle, "reserve_low_cpus": args.reserve,
        "mem_per_job_gb": args.mem_per_job_gb, "smoke": args.smoke,
        "variants": args.variants, "environments": args.envs, "seeds": args.seeds,
        "resolved_settings": {
            "initialisation": {
                "no_entropy": {"margin_loss_weight": 1.0, "safe_action_entropy_weight": 0.0,
                               "stopping_criterion": "target_margin", "target_margin": 2.0},
                "ce_only": {"margin_loss_weight": 0.0, "safe_action_entropy_weight": 0.0,
                            "stopping_criterion": "any_safe_feasibility",
                            "rashomon_multi_label_mode": "any"},
            },
            "timesteps": {
                environment: (8 if args.smoke else suite.environment_defaults(environment).total_timesteps)
                for environment in args.envs
            },
            "enforcement_frequency": {
                environment: suite.environment_defaults(environment).frequency
                for environment in args.envs
            },
            "adaptive_lid": {"region_refresh": "adaptive", "directional": True,
                             "iterations_per_lid": 1 if args.smoke else 200,
                             "objective": "weighted_width", "region_mode": "replace",
                             "stop_when_proposal_contained": True},
            "full_state_lid_batches": True, "full_state_certificates": True,
            "threads_per_job": 1, "nice": args.nice,
        },
        "canonical_controls": controls, "prepared_bases": bases,
        "assignment": {
            host.name: [
                {"cpu": item.cpu, "unit": item.unit.key, "variant": item.unit.variant,
                 "environment": item.unit.environment, "seed": item.unit.seed}
                for item in host.assignments
            ] for host in used
        },
    }
    manifest_path = dispatch_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    started = [host.name for host in used if launch_host(
        host, dispatch_dir=dispatch_dir, domain=args.domain, timeout=args.ssh_timeout
    )]
    atomic_json(dispatch_dir / "launch_result.json", {
        "created_at": utc_now(), "planned_hosts": [host.name for host in used],
        "started_hosts": started,
    })
    print(f"started {len(started)}/{len(used)} hosts; immutable manifest: {manifest_path}")
    return 0 if len(started) == len(used) else 1


def latest_manifest(output_root: Path) -> Path | None:
    paths = sorted((output_root / "_policy_initialisation_dispatch").glob("*/manifest.json"))
    return paths[-1] if paths else None


def status_main(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    bases_ready = 0
    for variant in args.variants:
        for environment in args.envs:
            try:
                validate_base(output_root, variant, environment)
                state = "ready"
                bases_ready += 1
            except FileNotFoundError:
                state = "missing"
            except ValueError as exc:
                state = f"FAILED ({exc})"
            print(f"base  {variant:<10} {environment:<20} {state}")
    completed = 0
    for variant in args.variants:
        for environment in args.envs:
            count = sum(
                valid_metrics(output_root / variant / ARCHITECTURE / environment / f"seed{seed}" / "metrics.json")
                for seed in args.seeds
            )
            completed += count
            print(f"train {variant:<10} {environment:<20} {count}/{len(args.seeds)}")
    print(f"bases: {bases_ready}/{len(args.variants) * len(args.envs)}")
    print(f"training: {completed}/{len(args.variants) * len(args.envs) * len(args.seeds)}")
    manifest = latest_manifest(output_root)
    if manifest:
        print(f"latest dispatch: {manifest}")
        if args.check_hosts:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for host in sorted(payload["assignment"]):
                # Anchor on the per-job wrapper: exactly one lives for the whole
                # job in both phases. The driver runs as "python -" (heredoc) and
                # only spawns a literal train_pspo.py during training, so matching
                # the stage scripts reports 0 for healthy base-phase jobs.
                remote = 'pgrep -c -u $USER -f "[r]un_pspo_one_env" || true'
                result = subprocess.run(
                    ssh_command(host, remote, domain=args.domain), capture_output=True,
                    text=True, timeout=args.ssh_timeout,
                )
                value = result.stdout.strip() if result.returncode == 0 else "ssh-failed"
                print(f"{host:<10} processes={value}")
    return 0


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.replace(" ", "").split(",") if item]
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError("expected distinct non-negative comma-separated integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="measure exact per-core idle capacity")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--hosts", default=None)
    probe.add_argument("--families", default=DEFAULT_FAMILIES)
    probe.add_argument("--source-root", type=Path, default=REPO)
    probe.add_argument("--domain", default="doc.ic.ac.uk")
    probe.add_argument("--minimum-idle", type=float, default=90.0)
    probe.add_argument("--reserve", type=int, default=2)
    probe.add_argument("--samples", type=int, default=5)
    probe.add_argument("--timeout", type=int, default=30)
    probe.add_argument("--fanout", type=int, default=60)
    probe.add_argument("--skip-auth-preflight", action="store_true")

    dispatch = sub.add_parser("dispatch", help="plan or launch one dependent wave")
    dispatch.add_argument("--hosts-tsv", type=Path, required=True)
    dispatch.add_argument("--phase", choices=("base", "train"), required=True)
    dispatch.add_argument("--source-root", type=Path, default=REPO)
    dispatch.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    dispatch.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    dispatch.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    dispatch.add_argument("--seeds", type=parse_csv_ints, default=list(range(10)))
    dispatch.add_argument("--mem-per-job-gb", type=float, default=2.0)
    dispatch.add_argument("--minimum-idle", type=float, default=90.0)
    dispatch.add_argument("--reserve", type=int, default=2)
    dispatch.add_argument("--max-hosts", type=int, default=0)
    dispatch.add_argument("--nice", type=int, default=5)
    dispatch.add_argument("--domain", default="doc.ic.ac.uk")
    dispatch.add_argument("--ssh-timeout", type=int, default=30)
    dispatch.add_argument("--revalidate-samples", type=int, default=2)
    dispatch.add_argument("--smoke", action="store_true")
    dispatch.add_argument("--allow-dirty", action="store_true")
    dispatch.add_argument("--launch", action="store_true")

    status = sub.add_parser("status", help="report base and seed completion")
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    status.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    status.add_argument("--seeds", type=parse_csv_ints, default=list(range(10)))
    status.add_argument("--check-hosts", action="store_true")
    status.add_argument("--domain", default="doc.ic.ac.uk")
    status.add_argument("--ssh-timeout", type=int, default=20)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("samples", "timeout", "fanout", "reserve", "revalidate_samples"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "minimum_idle") and not 0.0 <= args.minimum_idle <= 100.0:
        raise ValueError("--minimum-idle must lie in [0, 100]")
    if hasattr(args, "mem_per_job_gb") and args.mem_per_job_gb <= 0:
        raise ValueError("--mem-per-job-gb must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.command == "probe":
        return probe_main(args)
    if args.command == "dispatch":
        return dispatch_main(args)
    return status_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
