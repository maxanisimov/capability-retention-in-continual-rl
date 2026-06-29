"""Launch safe-policy-optimisation experiment pipelines by name."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from projects.safe_policy_optimisation.utils.config import (  # noqa: E402
    PIPELINES_FILE,
    REPO_ROOT,
    TASKS_FILE,
    apply_settings_to_namespace,
    cli_supplied_flags,
    compose_pipeline_settings,
    load_pipeline_registry,
    registry_source_file,
)
from projects.safe_policy_optimisation.stages import synthesise_shield  # noqa: E402


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    module: str
    default_task: str
    description: str


def available_pipelines(pipelines_file: Path = PIPELINES_FILE) -> dict[str, PipelineSpec]:
    pipelines = load_pipeline_registry(pipelines_file)
    specs: dict[str, PipelineSpec] = {}
    for name, payload in pipelines.items():
        if not isinstance(payload, dict):
            raise ValueError(f"{pipelines_file}: pipeline {name!r} must be a mapping.")
        specs[name] = PipelineSpec(
            name=name,
            module=str(payload["module"]),
            default_task=str(payload["default_task"]),
            description=str(payload.get("description", "")),
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a safe_policy_optimisation experiment pipeline. Unknown flags are "
            "passed through to the selected pipeline and override YAML settings."
        ),
    )
    parser.add_argument("--list-pipelines", action="store_true", help="List registered pipelines and exit.")
    parser.add_argument("--pipeline", help="Pipeline name to run.")
    parser.add_argument("--task", default=None, help="Task name from an environment-specific tasks.yaml.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=TASKS_FILE,
        help="Task registry YAML file or settings directory containing */tasks.yaml files.",
    )
    parser.add_argument(
        "--pipelines-file",
        type=Path,
        default=PIPELINES_FILE,
        help="Pipeline registry YAML file or settings directory containing */pipelines.yaml files.",
    )
    parser.add_argument("--settings-file", type=Path, default=None, help="Legacy flat YAML settings override.")
    parser.add_argument(
        "--force-shield-synthesis",
        action="store_true",
        help="Re-synthesise the configured shield even when shield_q.pt already exists.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the pipeline output directory.")
    parser.add_argument("--run-id", default=None, help="Override the run identifier.")
    return parser


def list_pipelines(*, pipelines_file: Path = PIPELINES_FILE) -> None:
    for name, spec in sorted(available_pipelines(pipelines_file).items()):
        print(f"{name}: {spec.description}")


def _load_pipeline_module(spec: PipelineSpec) -> Any:
    return importlib.import_module(spec.module)


def _pipeline_argv(
    *,
    output_dir: Path | None,
    run_id: str | None,
    passthrough: list[str],
) -> list[str]:
    argv: list[str] = []
    if output_dir is not None:
        argv.extend(["--output-dir", str(output_dir)])
    if run_id is not None:
        argv.extend(["--run-id", run_id])
    argv.extend(passthrough)
    return argv


def _synthesise_shield_if_needed(
    settings: dict[str, Any],
    *,
    task_settings: dict[str, Any],
    force: bool = False,
) -> Path:
    shield_path = Path(settings["shield_path"])
    if not shield_path.is_absolute():
        shield_path = REPO_ROOT / shield_path
    if shield_path.exists() and not force:
        print(f"Using existing shield: {shield_path}")
        return shield_path

    env_kwargs = task_settings.get("env_kwargs")
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, dict):
        raise ValueError("Task env_kwargs must be a mapping or null for shield synthesis.")
    constraint_kwargs = settings.get("constraint_kwargs")
    if constraint_kwargs is None:
        constraint_kwargs = {}
    if not isinstance(constraint_kwargs, dict):
        raise ValueError("constraint_kwargs must be a mapping for shield synthesis.")

    argv = [
        "--env",
        str(settings["env_id"]),
        "--task",
        str(task_settings.get("shield_task", settings["task"])),
        "--seed",
        str(settings.get("seed", 0)),
        "--constraint",
        str(settings.get("constraint", "PCTL")),
        "--constraint-kwargs",
        json.dumps(constraint_kwargs),
        "--theta",
        str(settings.get("theta", 1e-10)),
        "--max-vi-steps",
        str(settings.get("max_vi_steps", 1000)),
        "--init-safety-bound",
        str(settings.get("init_safety_bound", 0.5)),
        "--granularity",
        str(settings.get("granularity", 20)),
        "--unsafe-cost-threshold",
        str(settings.get("unsafe_cost_threshold", 0.5)),
        "--env-kwargs",
        json.dumps(env_kwargs),
        "--output-dir",
        str(shield_path.parent),
    ]
    if settings.get("max_episode_steps") is not None:
        argv.extend(["--max-episode-steps", str(settings["max_episode_steps"])])

    print(f"Synthesising shield before training: {shield_path}")
    result_path = synthesise_shield.run(synthesise_shield.build_parser().parse_args(argv))
    if Path(result_path) != shield_path:
        raise RuntimeError(f"Shield synthesis wrote {result_path}, expected {shield_path}")
    return shield_path


def run_pipeline(
    pipeline: str,
    *,
    task: str | None = None,
    tasks_file: Path = TASKS_FILE,
    pipelines_file: Path = PIPELINES_FILE,
    settings_file: Path | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    passthrough: list[str] | None = None,
    force_shield_synthesis: bool = False,
) -> dict[str, Any]:
    pipelines = available_pipelines(pipelines_file)
    if pipeline not in pipelines:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Available: {', '.join(sorted(pipelines))}")

    spec = pipelines[pipeline]
    module = _load_pipeline_module(spec)
    argv = _pipeline_argv(
        output_dir=output_dir,
        run_id=run_id,
        passthrough=list(passthrough or []),
    )
    if task is not None:
        argv.extend(["--task", task])
    if settings_file is not None:
        argv.extend(["--settings-file", str(settings_file)])
    args = module.build_parser().parse_args(argv)
    explicit_flags = cli_supplied_flags(argv)
    if settings_file is not None and hasattr(module, "apply_training_settings"):
        args = module.apply_training_settings(args, explicit_flags=cli_supplied_flags(argv))
    else:
        settings, _pipeline_settings, task_settings = compose_pipeline_settings(
            pipeline,
            task_name=task,
            tasks_file=tasks_file,
            pipelines_file=pipelines_file,
        )
        selected_task = str(settings["task"])
        pipeline_source = registry_source_file(
            pipelines_file,
            filename="pipelines.yaml",
            root_key="pipelines",
            name=pipeline,
        )
        task_source = registry_source_file(
            tasks_file,
            filename="tasks.yaml",
            root_key="tasks",
            name=selected_task,
        )
        args = apply_settings_to_namespace(
            args,
            settings,
            settings_file=pipeline_source,
            explicit_flags=explicit_flags,
        )
        args.training_task_settings_file = task_source
        args.training_pipeline_settings_file = pipeline_source
        _synthesise_shield_if_needed(
            settings,
            task_settings=task_settings,
            force=force_shield_synthesis,
        )
    return module.run(args)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args, passthrough = parser.parse_known_args(raw_argv)
    if args.list_pipelines:
        list_pipelines(pipelines_file=args.pipelines_file)
        return 0
    if args.pipeline is None:
        parser.error("--pipeline is required unless --list-pipelines is used")

    run_pipeline(
        args.pipeline,
        task=args.task,
        tasks_file=args.tasks_file,
        pipelines_file=args.pipelines_file,
        settings_file=args.settings_file,
        output_dir=args.output_dir,
        run_id=args.run_id,
        passthrough=passthrough,
        force_shield_synthesis=args.force_shield_synthesis,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
