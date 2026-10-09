"""Launch PSPO on idle, disjoint CPU cores."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projects.safe_policy_optimisation.utils.pspo_defaults import (
    BC_INITIALISATION_OBJECTIVE,
    BC_MARGIN_LOSS_WEIGHT,
    BC_MARGIN_MODE,
    BC_MAX_UNSAFE_MASS,
    BC_MIN_SAFE_ACTION_ENTROPY,
    BC_SAFE_ACTION_ENTROPY_WEIGHT,
    BC_SAFE_ACTION_UNIFORMITY_WEIGHT,
    BC_TARGET_MARGIN,
    BC_UNSAFE_MASS_TARGET,
    RASHOMON_MULTI_LABEL_MODE,
    RASHOMON_N_ITERS,
    RASHOMON_OBJECTIVE,
    RASHOMON_SURROGATE,
    environment_defaults,
)
from projects.safe_policy_optimisation.utils.pspo_launcher import (
    parse_cpu_ids,
)

REPO = Path(__file__).resolve().parents[3]
RUNS_ROOT = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
)
ONE_ENV_LAUNCHER = (
    REPO
    / "projects/safe_policy_optimisation/scripts/run_pspo_one_env.sh"
)
DEFAULT_ENVS = (
    "media_streaming",
    "colour_bomb",
    "colour_bomb_v2",
    "bridge_crossing",
    "bridge_crossing_v2",
    "mini_pacman",
)
ARCHITECTURES = {
    "one_hidden": {"n_hidden": 1, "hidden_dim": 64, "activation": "Tanh"},
    "two_hidden": {"n_hidden": 2, "hidden_dim": 64, "activation": "Tanh"},
}
PIPELINES = {
    "media_streaming": "paper_2503_07671_media_streaming",
    "colour_bomb": "paper_2503_07671_colour_bomb",
    "colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
    "bridge_crossing": "paper_2503_07671_bridge_crossing",
    "bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
    "mini_pacman": "paper_2503_07671_minipacman",
}


def parse_mpstat_idle(output: str) -> dict[int, float]:
    """Extract average per-core idle percentages from ``mpstat -P ALL``."""

    idle: dict[int, float] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] != "Average:" or not fields[1].isdigit():
            continue
        try:
            idle[int(fields[1])] = float(fields[-1])
        except ValueError:
            continue
    if not idle:
        raise ValueError("mpstat output contains no average per-core idle measurements")
    return idle


def sample_cpu_idle(sample_seconds: int) -> dict[int, float]:
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    try:
        completed = subprocess.run(
            ["mpstat", "-P", "ALL", "1", str(sample_seconds)],
            cwd=REPO,
            env={**os.environ, "LC_ALL": "C"},
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("mpstat is required for automatic idle-core selection") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"mpstat failed: {exc.stderr.strip()}") from exc
    return parse_mpstat_idle(completed.stdout)


def select_idle_cpus(
    idle_by_cpu: dict[int, float],
    *,
    required: int,
    minimum_idle: float,
    allowed_cpus: set[int],
) -> list[int]:
    """Choose the most idle allowed cores, with deterministic tie-breaking."""

    eligible = [
        (cpu, idle)
        for cpu, idle in idle_by_cpu.items()
        if cpu in allowed_cpus and idle >= minimum_idle
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < required:
        raise RuntimeError(
            f"Need {required} cores at least {minimum_idle:.1f}% idle, but only "
            f"{len(eligible)} qualify. No jobs were launched."
        )
    return [cpu for cpu, _ in eligible[:required]]


def safety_demo_sizes(environments: list[str]) -> dict[str, int]:
    """Read each shield and return its complete demonstration-dataset size."""

    from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
        load_shield_mask,
        make_safe_behaviour_payload,
    )
    from projects.safe_policy_optimisation.utils.config import compose_pipeline_settings

    sizes: dict[str, int] = {}
    for environment in environments:
        cfg, _, _ = compose_pipeline_settings(PIPELINES[environment])
        shield_path = Path(cfg["shield_path"])
        if not shield_path.is_absolute():
            shield_path = REPO / shield_path
        mask = load_shield_mask(shield_path)
        _, metadata = make_safe_behaviour_payload(mask)
        sizes[environment] = int(metadata["dataset_size"])
    return sizes


def build_launch_environment(
    *,
    environment: str,
    seeds: list[int],
    cpu_ids: list[int],
    architecture: str,
    run_name: str,
    n_iters: int,
    dry_run: bool,
    adaptive_freq: str | None = None,
    directional: bool = True,
    rashomon_objective: str = RASHOMON_OBJECTIVE,
    safe_region_shape: str = "orthotope",
    segment_tolerance: float = 1e-3,
    segment_splits: int = 4,
    segment_max_splits: int = 8,
    verify_first: bool = False,
    region_refresh: str = "adaptive",
    audit_candidates_exactly: bool = False,
    bc_margin_loss_weight: float = BC_MARGIN_LOSS_WEIGHT,
    bc_safe_action_entropy_weight: float = BC_SAFE_ACTION_ENTROPY_WEIGHT,
    bc_min_safe_action_entropy: float = BC_MIN_SAFE_ACTION_ENTROPY,
    bc_initialisation_objective: str = BC_INITIALISATION_OBJECTIVE,
    bc_unsafe_mass_target: float = BC_UNSAFE_MASS_TARGET,
    bc_max_unsafe_mass: float = BC_MAX_UNSAFE_MASS,
    bc_safe_action_uniformity_weight: float = BC_SAFE_ACTION_UNIFORMITY_WEIGHT,
) -> dict[str, str]:
    """Build the exact one-environment launcher configuration."""

    task_defaults = environment_defaults(environment)

    return {
        **os.environ,
        "ENV_NAME": environment,
        "SEEDS": " ".join(str(seed) for seed in seeds),
        "CPU_IDS": ",".join(str(cpu) for cpu in cpu_ids),
        "ARCHITECTURE": architecture,
        "REGION_MODE": "replace",
        "RUN_NAME": run_name,
        "RASHOMON_MULTI_LABEL_MODE": RASHOMON_MULTI_LABEL_MODE,
        "RASHOMON_SURROGATE": RASHOMON_SURROGATE,
        "RASHOMON_OBJECTIVE": rashomon_objective,
        "SAFE_REGION_SHAPE": safe_region_shape,
        "SEGMENT_TOLERANCE": str(segment_tolerance),
        "SEGMENT_SPLITS": str(segment_splits),
        "SEGMENT_MAX_SPLITS": str(segment_max_splits),
        "VERIFY_FIRST": "true" if verify_first else "false",
        "REGION_REFRESH": region_refresh,
        "AUDIT_CANDIDATES_EXACTLY": "1" if audit_candidates_exactly else "0",
        "RASHOMON_BATCH_SIZE": "auto",
        "RASHOMON_CERTIFICATE_SAMPLES": "all",
        "RASHOMON_N_ITERS": str(n_iters),
        "BC_TARGET_MARGIN": str(BC_TARGET_MARGIN),
        "BC_MARGIN_LOSS_WEIGHT": str(bc_margin_loss_weight),
        "BC_SAFE_ACTION_ENTROPY_WEIGHT": str(bc_safe_action_entropy_weight),
        "BC_MIN_SAFE_ACTION_ENTROPY": str(bc_min_safe_action_entropy),
        "BC_INITIALISATION_OBJECTIVE": bc_initialisation_objective,
        "BC_UNSAFE_MASS_TARGET": str(bc_unsafe_mass_target),
        "BC_MAX_UNSAFE_MASS": str(bc_max_unsafe_mass),
        "BC_SAFE_ACTION_UNIFORMITY_WEIGHT": str(
            bc_safe_action_uniformity_weight
        ),
        "DIRECTIONAL_RASHOMON_GROWTH": "1" if directional else "0",
        "ADAPTIVE_FREQ": adaptive_freq or task_defaults.frequency,
        "TOTAL_TIMESTEPS": str(task_defaults.total_timesteps),
        "STOP_WHEN_PROPOSAL_CONTAINED": "1",
        "SKIP_EXISTING": "1",
        "DRY_RUN": "1" if dry_run else "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch PSPO with directional Local Independence Domain "
            "(LID) growth and all-safe-logit semantics across the paper MASA "
            "environments."
        )
    )

    experiment = parser.add_argument_group("experiment selection")
    experiment.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    experiment.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    experiment.add_argument(
        "--run-name",
        default=None,
        help="Output run name. By default it is derived from --architecture.",
    )

    ppo = parser.add_argument_group("PPO update settings")
    ppo.add_argument(
        "--freq",
        default=None,
        help=(
            "Unified frequency override: update, rollout, once, or a positive "
            "rollout count. By default, use each environment's best setting "
            "(100 for MiniPacman; 1 for the others)."
        ),
    )

    initialisation = parser.add_argument_group("policy initialisation")
    initialisation.add_argument(
        "--architecture",
        choices=tuple(ARCHITECTURES),
        default="two_hidden",
        help="Policy architecture. Defaults to the original two-hidden-layer experiment.",
    )
    initialisation.add_argument(
        "--bc-initialisation-objective",
        choices=("margin", "safe_mass"),
        default=BC_INITIALISATION_OBJECTIVE,
        help="Use the legacy margin loss or the new probability-mass loss.",
    )
    initialisation.add_argument(
        "--bc-margin-loss-weight",
        type=float,
        default=BC_MARGIN_LOSS_WEIGHT,
        help="Weight of the all-safe logit-margin hinge; 0 selects CE-only fitting.",
    )
    initialisation.add_argument(
        "--bc-safe-action-entropy-weight",
        type=float,
        default=BC_SAFE_ACTION_ENTROPY_WEIGHT,
        help=(
            "Weight of the safe-action conditional-entropy regularizer used for "
            "base-policy fitting. Enabled by default with the best-known weight."
        ),
    )
    initialisation.add_argument(
        "--bc-min-safe-action-entropy",
        type=float,
        default=BC_MIN_SAFE_ACTION_ENTROPY,
        help="Required minimum normalized safe-action entropy when enabled.",
    )
    initialisation.add_argument(
        "--bc-unsafe-mass-target",
        type=float,
        default=BC_UNSAFE_MASS_TARGET,
        help="Finite unsafe-mass target for the safe_mass objective.",
    )
    initialisation.add_argument(
        "--bc-max-unsafe-mass",
        type=float,
        default=BC_MAX_UNSAFE_MASS,
        help="Per-state unsafe-mass threshold for stopping safe_mass fitting.",
    )
    initialisation.add_argument(
        "--bc-safe-action-uniformity-weight",
        type=float,
        default=BC_SAFE_ACTION_UNIFORMITY_WEIGHT,
        help="Weight of the uniform-safe-action KL in the safe_mass objective.",
    )

    lid = parser.add_argument_group("LID settings")
    lid.add_argument(
        "--region-refresh",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Recompute LIDs adaptively or construct one fixed LID before PPO.",
    )
    lid.add_argument(
        "--audit-candidates-exactly",
        action="store_true",
        help="Audit region-first candidates exactly without changing decisions.",
    )
    lid.add_argument(
        "--lid-n-iters",
        dest="rashomon_n_iters",
        metavar="ITERATIONS",
        type=int,
        default=RASHOMON_N_ITERS,
        help="Maximum optimization iterations used to construct each LID.",
    )
    lid.add_argument(
        "--directional",
        choices=("true", "false"),
        default="true",
        help="Whether to grow each LID towards the proposed PPO update.",
    )
    lid.add_argument(
        "--lid-objective",
        dest="rashomon_objective",
        choices=("weighted_width", "projection_distance"),
        default=RASHOMON_OBJECTIVE,
        help="Objective used to grow each certified LID.",
    )
    lid.add_argument(
        "--safe-region-shape",
        choices=("orthotope", "zonotope", "segment"),
        default="orthotope",
        help=(
            "Certified region geometry. 'segment' certifies the longest safe "
            "step along the proposed update (rank-one zonotope)."
        ),
    )
    lid.add_argument(
        "--segment-tolerance",
        type=float,
        default=1e-3,
        help="Bisection tolerance on the certified step length (segment only).",
    )
    lid.add_argument(
        "--segment-splits",
        type=int,
        default=4,
        help="Sub-segments per certificate evaluation (segment only).",
    )
    lid.add_argument(
        "--segment-max-splits",
        type=int,
        default=8,
        help="Split cap when refining a failed segment (segment only).",
    )
    lid.add_argument(
        "--verify-first",
        choices=("true", "false"),
        default="false",
        help=(
            "Whether to verify a proposed policy before constructing its LID. "
            "Defaults to LID-first behavior (false)."
        ),
    )

    # Keep historical spellings accepted so existing launch commands continue to
    # work, but omit them from --help in favour of the literature-aligned LID names.
    parser.add_argument(
        "--rashomon-n-iters",
        dest="legacy_rashomon_n_iters",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rashomon-objective",
        dest="legacy_rashomon_objective",
        choices=("weighted_width", "projection_distance"),
        default=None,
        help=argparse.SUPPRESS,
    )

    execution = parser.add_argument_group("CPU allocation and execution")
    execution.add_argument(
        "--cpu-ids",
        default=None,
        help="Optional explicit CPU list/ranges; otherwise cores are sampled with mpstat.",
    )
    execution.add_argument("--minimum-idle", type=float, default=90.0)
    execution.add_argument("--sample-seconds", type=int, default=5)
    execution.add_argument("--dry-run", action="store_true")
    return parser


def _option_was_supplied(argv: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in argv
    )


def parse_launcher_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse launcher arguments and resolve temporarily supported legacy aliases."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    aliases = (
        (
            "--rashomon-n-iters",
            "--lid-n-iters",
            "legacy_rashomon_n_iters",
            "rashomon_n_iters",
        ),
        (
            "--rashomon-objective",
            "--lid-objective",
            "legacy_rashomon_objective",
            "rashomon_objective",
        ),
    )
    for legacy_option, replacement, legacy_dest, canonical_dest in aliases:
        legacy_value = getattr(args, legacy_dest)
        if legacy_value is not None:
            if _option_was_supplied(raw_argv, replacement):
                parser.error(f"{legacy_option} cannot be combined with {replacement}")
            print(
                f"warning: {legacy_option} is deprecated; use {replacement} instead. "
                "It will be removed in the next CLI-breaking cleanup.",
                file=sys.stderr,
            )
            setattr(args, canonical_dest, legacy_value)
        delattr(args, legacy_dest)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.envs or len(set(args.envs)) != len(args.envs):
        raise SystemExit("--envs must contain distinct environment names")
    unknown = sorted(set(args.envs) - set(DEFAULT_ENVS))
    if unknown:
        raise SystemExit(
            f"Unsupported environments {unknown}; expected choices from {DEFAULT_ENVS}"
        )
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must contain distinct values")
    if args.rashomon_n_iters <= 0:
        raise SystemExit("--lid-n-iters must be positive")
    if (
        not math.isfinite(args.bc_margin_loss_weight)
        or args.bc_margin_loss_weight < 0.0
    ):
        raise SystemExit("--bc-margin-loss-weight must be finite and non-negative")
    if (
        not math.isfinite(args.bc_safe_action_entropy_weight)
        or args.bc_safe_action_entropy_weight < 0.0
    ):
        raise SystemExit(
            "--bc-safe-action-entropy-weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.bc_min_safe_action_entropy)
        or not 0.0 <= args.bc_min_safe_action_entropy <= 1.0
    ):
        raise SystemExit("--bc-min-safe-action-entropy must lie in [0, 1]")
    if (
        not math.isfinite(args.bc_unsafe_mass_target)
        or not 0.0 < args.bc_unsafe_mass_target < 0.5
    ):
        raise SystemExit("--bc-unsafe-mass-target must lie strictly between 0 and 0.5")
    if (
        not math.isfinite(args.bc_max_unsafe_mass)
        or not 0.0 <= args.bc_max_unsafe_mass < 0.5
    ):
        raise SystemExit("--bc-max-unsafe-mass must lie in [0, 0.5)")
    if args.bc_unsafe_mass_target > args.bc_max_unsafe_mass:
        raise SystemExit("--bc-unsafe-mass-target must not exceed --bc-max-unsafe-mass")
    if (
        not math.isfinite(args.bc_safe_action_uniformity_weight)
        or args.bc_safe_action_uniformity_weight < 0.0
    ):
        raise SystemExit(
            "--bc-safe-action-uniformity-weight must be finite and non-negative"
        )
    if not 0.0 <= args.minimum_idle <= 100.0:
        raise SystemExit("--minimum-idle must lie in [0, 100]")
    if args.freq is not None:
        normalized = str(args.freq).strip().lower()
        if normalized not in {"update", "rollout", "once"}:
            try:
                interval = int(normalized)
            except ValueError as exc:
                raise SystemExit(
                    "--freq must be update, rollout, once, or a positive integer"
                ) from exc
            if interval <= 0:
                raise SystemExit("numeric --freq must be positive")
    if args.freq == "once" and args.directional != "false":
        raise SystemExit("--freq once requires --directional false")
    if args.freq == "once" and args.verify_first != "false":
        raise SystemExit("--freq once requires --verify-first false")
    if args.region_refresh == "fixed" and args.directional != "false":
        raise SystemExit("--region-refresh fixed requires --directional false")
    if args.region_refresh == "fixed" and args.verify_first != "false":
        raise SystemExit("--region-refresh fixed requires --verify-first false")
    if args.audit_candidates_exactly and args.verify_first != "false":
        raise SystemExit("--audit-candidates-exactly is only valid for region-first runs")
    if (
        args.rashomon_objective == "projection_distance"
        and args.directional != "true"
    ):
        raise SystemExit("--lid-objective projection_distance requires --directional true")


def main(argv: list[str] | None = None) -> int:
    args = parse_launcher_args(argv)
    _validate_args(args)
    environments = list(args.envs)
    seeds = list(args.seeds)
    architecture = str(args.architecture)
    architecture_settings = ARCHITECTURES[architecture]
    growth_tag = "directional" if args.directional == "true" else "nondirectional"
    default_run_name = (
        f"pspo_{architecture}_{growth_tag}_replace_all_margin2_200iters"
    )
    if args.rashomon_objective != "weighted_width":
        default_run_name += f"_{args.rashomon_objective}"
    if args.verify_first == "true":
        default_run_name += "_verify_first"
    if args.bc_safe_action_entropy_weight > 0.0:
        entropy_weight_tag = format(args.bc_safe_action_entropy_weight, "g").replace(
            ".", "p"
        )
        min_entropy_tag = format(args.bc_min_safe_action_entropy, "g").replace(
            ".", "p"
        )
        default_run_name += f"_safe_entropy_w{entropy_weight_tag}_min{min_entropy_tag}"
    if args.bc_margin_loss_weight == 0.0:
        default_run_name += "_ce_only"
    if args.bc_initialisation_objective == "safe_mass":
        epsilon_tag = format(args.bc_unsafe_mass_target, "g").replace(".", "p")
        default_run_name += f"_safe_mass_eps{epsilon_tag}"
    run_name = args.run_name or default_run_name
    required = len(environments) * len(seeds)
    allowed_cpus = set(os.sched_getaffinity(0))

    idle_by_cpu: dict[int, float] | None = None
    if args.cpu_ids:
        selected_cpus = parse_cpu_ids(args.cpu_ids)
        if len(selected_cpus) != required:
            raise SystemExit(
                f"--cpu-ids supplies {len(selected_cpus)} CPUs; exactly {required} are required"
            )
        unavailable = sorted(set(selected_cpus) - allowed_cpus)
        if unavailable:
            raise SystemExit(f"CPUs outside this process's affinity: {unavailable}")
    else:
        try:
            idle_by_cpu = sample_cpu_idle(args.sample_seconds)
            selected_cpus = select_idle_cpus(
                idle_by_cpu,
                required=required,
                minimum_idle=float(args.minimum_idle),
                allowed_cpus=allowed_cpus,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    allocation = {
        environment: selected_cpus[index * len(seeds):(index + 1) * len(seeds)]
        for index, environment in enumerate(environments)
    }
    dataset_sizes = safety_demo_sizes(environments)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "environments": environments,
        "seeds": seeds,
        "cpu_allocation": allocation,
        "cpu_idle_percent": idle_by_cpu,
        "minimum_idle_percent": float(args.minimum_idle),
        "settings": {
            "architecture": architecture,
            **architecture_settings,
            "region_update_mode": "replace",
            "verify_first": args.verify_first == "true",
            "region_refresh": args.region_refresh,
            "audit_candidates_exactly": bool(args.audit_candidates_exactly),
            "directional_rashomon_growth": args.directional == "true",
            "stop_when_proposal_contained": args.directional == "true",
            "rashomon_n_iters": int(args.rashomon_n_iters),
            "frequency": {
                environment: args.freq or environment_defaults(environment).frequency
                for environment in environments
            },
            "total_timesteps": {
                environment: environment_defaults(environment).total_timesteps
                for environment in environments
            },
            "rashomon_multi_label_mode": RASHOMON_MULTI_LABEL_MODE,
            "rashomon_surrogate": RASHOMON_SURROGATE,
            "rashomon_objective": str(args.rashomon_objective),
            "bc_margin_mode": BC_MARGIN_MODE,
            "bc_target_margin": BC_TARGET_MARGIN,
            "bc_margin_loss_weight": float(args.bc_margin_loss_weight),
            "bc_safe_action_entropy_weight": float(args.bc_safe_action_entropy_weight),
            "bc_min_safe_action_entropy": float(args.bc_min_safe_action_entropy),
            "bc_initialisation_objective": args.bc_initialisation_objective,
            "bc_unsafe_mass_target": float(args.bc_unsafe_mass_target),
            "bc_max_unsafe_mass": float(args.bc_max_unsafe_mass),
            "bc_safe_action_uniformity_weight": float(
                args.bc_safe_action_uniformity_weight
            ),
            "safety_demo_sizes": dataset_sizes,
            "rashomon_batch_size": dataset_sizes,
            "certificate_samples": dataset_sizes,
        },
    }

    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for environment in environments:
            env = build_launch_environment(
                environment=environment,
                seeds=seeds,
                cpu_ids=allocation[environment],
                architecture=architecture,
                run_name=run_name,
                n_iters=args.rashomon_n_iters,
                dry_run=True,
                adaptive_freq=args.freq,
                directional=args.directional == "true",
                rashomon_objective=args.rashomon_objective,
                safe_region_shape=args.safe_region_shape,
                segment_tolerance=args.segment_tolerance,
                segment_splits=args.segment_splits,
                segment_max_splits=args.segment_max_splits,
                verify_first=args.verify_first == "true",
                region_refresh=args.region_refresh,
                audit_candidates_exactly=args.audit_candidates_exactly,
                bc_margin_loss_weight=args.bc_margin_loss_weight,
                bc_safe_action_entropy_weight=args.bc_safe_action_entropy_weight,
                bc_min_safe_action_entropy=args.bc_min_safe_action_entropy,
                bc_initialisation_objective=args.bc_initialisation_objective,
                bc_unsafe_mass_target=args.bc_unsafe_mass_target,
                bc_max_unsafe_mass=args.bc_max_unsafe_mass,
                bc_safe_action_uniformity_weight=(
                    args.bc_safe_action_uniformity_weight
                ),
            )
            completed = subprocess.run(
                ["bash", str(ONE_ENV_LAUNCHER)],
                cwd=REPO,
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                return completed.returncode
        return 0

    log_dir = RUNS_ROOT / run_name / "_orchestrator"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for environment in environments:
        env = build_launch_environment(
            environment=environment,
            seeds=seeds,
            cpu_ids=allocation[environment],
            architecture=architecture,
            run_name=run_name,
            n_iters=args.rashomon_n_iters,
            dry_run=False,
            adaptive_freq=args.freq,
            directional=args.directional == "true",
            rashomon_objective=args.rashomon_objective,
            safe_region_shape=args.safe_region_shape,
            segment_tolerance=args.segment_tolerance,
            segment_splits=args.segment_splits,
            segment_max_splits=args.segment_max_splits,
            verify_first=args.verify_first == "true",
            region_refresh=args.region_refresh,
            audit_candidates_exactly=args.audit_candidates_exactly,
            bc_margin_loss_weight=args.bc_margin_loss_weight,
            bc_safe_action_entropy_weight=args.bc_safe_action_entropy_weight,
            bc_min_safe_action_entropy=args.bc_min_safe_action_entropy,
            bc_initialisation_objective=args.bc_initialisation_objective,
            bc_unsafe_mass_target=args.bc_unsafe_mass_target,
            bc_max_unsafe_mass=args.bc_max_unsafe_mass,
            bc_safe_action_uniformity_weight=(
                args.bc_safe_action_uniformity_weight
            ),
        )
        log_handle = (log_dir / f"{environment}.log").open("w")
        process = subprocess.Popen(
            ["bash", str(ONE_ENV_LAUNCHER)],
            cwd=REPO,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((environment, process, log_handle))
        print(f"launched {environment}: pid={process.pid}, cores={allocation[environment]}")

    failed: list[tuple[str, int]] = []
    for environment, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        print(f"finished {environment}: rc={returncode}", flush=True)
        if returncode != 0:
            failed.append((environment, returncode))
    if failed:
        print(f"failed environment launchers: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
