#!/usr/bin/env python3
"""Recover timing for the exact main reward/safety comparison, without reruns.

Process elapsed, instrumented stage/learn timers, and TensorBoard logged
intervals are distinct metrics. Missing observations are never imputed.
TensorBoard files are inspected at their ends rather than loading millions of
per-step events; TFRecord checksums establish record boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_final_policy_table import METHODS  # noqa: E402
from google.protobuf.message import DecodeError  # noqa: E402
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENTS,
    EXPECTED_SEEDS,
)
from tensorboard.compat.proto.event_pb2 import Event  # noqa: E402
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import (  # noqa: E402
    masked_crc32c,
)

DEFAULT_OUTPUT = (
    REPO / "projects/safe_policy_optimisation/results/training_time_main_baselines"
)
BASELINE_RE = re.compile(r"^\[\d+/\d+\] ok ([^/]+)/seed(\d+)/([a-z_]+) \((\d+)s\)\s*$")
PSPO_RE = re.compile(r"^ok seed(\d+) core=(\d+) (\d+)s\s*$")
GROUPS = {
    "ppo_policy": "ppo",
    "ppo_shield": "shielded",
    "cpo": "cpo",
    "ppo_lagrangian": "baselines_lag",
    "ppo_pid_lagrangian": "baselines_lag",
}
METRICS = {
    "rl_process_s": "Launcher RL process elapsed",
    "total_s": "Initialisation + RL process (cold single-seed execution)",
    "logged_interval_s": "Partial TensorBoard logged interval",
    "initialisation_s": "Shared policy initialisation",
    "training_loop_s": "Instrumented learn() interval",
    "rl_stage_s": "Instrumented RL stage",
    "pipeline_stage_s": "Recorded pipeline worker stage (distinct from process elapsed)",
    "lid_s": "Nested LID construction (not additive)",
    "verification_s": "Nested verification (not additive)",
    "projection_s": "Nested projection (not additive)",
    "finalisation_s": "Separately instrumented finalisation",
}


@lru_cache(maxsize=128)
def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def reference(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def seconds(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid duration: {value!r}")
    return number


def launcher_paths(environment, adaptive: bool) -> list[Path]:
    """Explicit cohort-bound sources; do not search unrelated retries/cohorts."""
    if adaptive:
        root = environment.adaptive_root.parents[1] / "_orchestrator"
        names = (
            ["extend_remaining_20260826.log", "screen.log"]
            if environment.key == "bridge_crossing_v2"
            else ["screen.log"]
            if environment.key == "mini_pacman"
            else [f"{environment.key}.log"]
        )
        return [root / name for name in names]
    if environment.key == "bridge_crossing_v2":
        return sorted(
            (environment.baseline_root.parent / "_remote_launch_logs").glob("*.log")
        )
    if environment.key == "mini_pacman":
        return sorted((environment.baseline_root.parent / "_dispatch").glob("*__*.log"))
    return [
        REPO
        / "outputs"
        / f"_sweeps_2hidden_{group}_launch_logs"
        / f"{environment.key}.log"
        for group in ("ppo", "shielded")
    ]


def read_completions(environment, adaptive: bool) -> dict[tuple[int, str], list[dict]]:
    records: dict[tuple[int, str], list[dict]] = {}
    for path in launcher_paths(environment, adaptive):
        if not path.is_file():
            continue
        content = path.read_text()
        if not adaptive and environment.key in ("bridge_crossing_v2", "mini_pacman"):
            if f"smoke={environment.nominal_budget}" not in content:
                raise ValueError(
                    f"Launcher budget does not match selected cohort: {path}"
                )
        if adaptive:
            # The standard orchestrator records its exact destination. Extension
            # logs belong to the selected cohort and are explicitly allowlisted.
            outputs = re.findall(r"^\S+: output (.+)$", content, re.MULTILINE)
            if outputs and str(environment.adaptive_root) not in outputs:
                raise ValueError(f"PSPO launcher destination mismatch: {path}")
        for line_number, line in enumerate(content.splitlines(), 1):
            match = (PSPO_RE if adaptive else BASELINE_RE).fullmatch(line)
            if not match:
                continue
            if adaptive:
                seed, core, elapsed = match.groups()
                group = "pspo"
                host = None
            else:
                env_key, seed, group, elapsed = match.groups()
                if env_key != environment.key:
                    continue
                core = None
                host = (
                    path.name.split("__")[0].split("-")[0]
                    if path.name.startswith(("ash", "yew"))
                    else None
                )
            records.setdefault((int(seed), group), []).append(
                {
                    "seconds": float(elapsed),
                    "source": f"{reference(path)}:{line_number}",
                    "host": host,
                    "core": core,
                }
            )
    return records


def decode_record(data: bytes, offset: int) -> tuple[Event, int]:
    if offset + 12 > len(data):
        raise ValueError("Truncated TFRecord header")
    length, crc = struct.unpack_from("<QI", data, offset)
    if masked_crc32c(data[offset : offset + 8]) != crc:
        raise ValueError("Invalid TFRecord header checksum")
    end = offset + 12 + length
    if end + 4 > len(data):
        raise ValueError("Truncated TFRecord payload")
    payload = data[offset + 12 : end]
    if masked_crc32c(payload) != struct.unpack_from("<I", data, end)[0]:
        raise ValueError("Invalid TFRecord payload checksum")
    event = Event.FromString(payload)
    if not math.isfinite(event.wall_time) or event.wall_time <= 0:
        raise ValueError("Invalid event wall timestamp")
    return event, end + 4


def event_endpoints(path: Path) -> dict:
    """Read writer start and a CRC-verified suffix, not filesystem timestamps.

    A corrupt/truncated suffix is unavailable, never repaired by choosing an
    earlier event. The max step is from the suffix; final-step alignment is
    required before a stream can be associated with the selected run.
    """
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("Missing writer-start event")
        length = struct.unpack_from("<Q", header)[0]
        if length > 4 * 1024 * 1024:
            raise ValueError("Oversized writer-start event")
        first, _ = decode_record(header + handle.read(length + 4), 0)
        handle.seek(0, 2)
        size = handle.tell()
        window = min(size, 65536)
        while True:
            handle.seek(size - window)
            tail = handle.read(window)
            boundary = None
            for offset in range(max(0, len(tail) - 15)):
                length, crc = struct.unpack_from("<QI", tail, offset)
                if not 0 < length <= len(tail) - offset - 16:
                    continue
                if masked_crc32c(tail[offset : offset + 8]) != crc:
                    continue
                # A verified length header establishes the first boundary.
                # Subsequent corruption must fail, not trigger a later start.
                boundary = offset
                break
            if boundary is not None:
                events = []
                while boundary < len(tail):
                    event, boundary = decode_record(tail, boundary)
                    events.append(event)
                end = max(event.wall_time for event in events)
                if end < first.wall_time:
                    raise ValueError("Event timestamps precede writer start")
                host_match = re.fullmatch(
                    r"events\.out\.tfevents\.\d+\.(.+?)\.\d+(?:\.\d+)?", path.name
                )
                return {
                    "start_wall_time": first.wall_time,
                    "end_wall_time": end,
                    "seconds": end - first.wall_time,
                    "last_step": max(event.step for event in events),
                    "source": reference(path),
                    "host": host_match.group(1) if host_match else None,
                }
            if window == size or window >= 4 * 1024 * 1024:
                raise ValueError("No complete TFRecord boundary in suffix")
            window = min(size, window * 2)


def select_event_stream(
    paths: list[Path], final_steps: int
) -> tuple[dict | None, str, list[dict]]:
    streams = []
    for path in paths:
        try:
            streams.append(event_endpoints(path))
        except (ValueError, OSError, DecodeError) as error:
            streams.append({"source": reference(path), "error": str(error)})
    candidates = [
        stream for stream in streams if stream.get("last_step") == final_steps
    ]
    if len(candidates) == 1:
        return candidates[0], "partial_logged_interval", streams
    return (
        None,
        "ambiguous_completed_streams" if candidates else "no_final_step_aligned_stream",
        streams,
    )


def add_pipeline_timing(
    row: dict, directory: Path, method, config: dict, summary: dict
) -> None:
    """Parent summaries may survive only for the last concurrently written group.

    Accept an embedded stage only if its exact summary, output, environment and
    budget match the selected method. A Lag/PID stage remains a joint observation.
    The worker timer includes module import/argument parsing and is not mixed with
    inside-run PSPO timers or full launcher process elapsed.
    """
    if method.adaptive:
        return
    path = directory.parent / "summary.json"
    parent = read_json(path)
    stage_key = "ppo_lagrangian" if method.key == "ppo_pid_lagrangian" else method.key
    stage = parent.get("stages", {}).get(stage_key)
    row["pipeline_stage_status"] = "missing_selected_stage_record"
    if not stage:
        return
    row["pipeline_stage_source"] = reference(path) + f"#stages.{stage_key}"
    if (
        parent.get("total_timesteps_budget") != row["nominal_timesteps"]
        or parent.get("env_id") != config.get("env_id")
        or parent.get("env_kwargs") != config.get("env_kwargs")
        or Path(stage.get("run_dir", "")).resolve() != directory.resolve()
        or stage.get("summary") != summary
    ):
        row["pipeline_stage_status"] = "record_does_not_match_selected_run"
        return
    elapsed = seconds(stage.get("elapsed_seconds"))
    start, end = stage.get("started_at"), stage.get("finished_at")
    if (
        elapsed is None
        or start is None
        or end is None
        or end < start
        or not math.isclose(end - start, elapsed, abs_tol=0.001)
        or (
            row.get("logged_start_wall_time") is not None
            and (
                row["logged_start_wall_time"] < start - 0.001
                or row["logged_end_wall_time"] > end + 0.001
            )
        )
    ):
        row["pipeline_stage_status"] = "invalid_or_unaligned_stage_interval"
        return
    row.update(
        pipeline_cpu_ids=stage.get("cpu_ids"),
        pipeline_device=parent.get("device"),
        pipeline_torch_num_threads=parent.get("torch_num_threads"),
    )
    if method.key in ("ppo_lagrangian", "ppo_pid_lagrangian"):
        row["combined_lag_pid_pipeline_stage_s"] = elapsed
        row["pipeline_stage_status"] = "combined_lag_pid_not_separable"
    else:
        row["pipeline_stage_s"] = elapsed
        row["pipeline_stage_status"] = "matched_instrumented_worker_stage"


def build_seed_row(environment, method, seed: int, completions: dict) -> dict:
    root = environment.adaptive_root if method.adaptive else environment.baseline_root
    directory = root / f"seed{seed}" / Path(method.relative_path).parent
    config = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    metrics = read_json(root / f"seed{seed}" / method.relative_path)
    if not config or not summary or not metrics:
        raise ValueError(f"Missing selected run records: {directory}")
    evaluated = metrics
    for key in method.section:
        evaluated = evaluated[key]
    if int(evaluated["eval_episodes"]) != 100:
        raise ValueError(f"Unexpected final evaluation protocol: {directory}")
    section = summary.get(method.key, summary)
    budget = config["total_timesteps"]
    budget = budget[method.key] if isinstance(budget, dict) else budget
    if int(budget) != environment.nominal_budget or int(config["seed"]) != seed:
        raise ValueError(f"Selected run seed/budget mismatch: {directory}")
    steps = int(section["final_timesteps"])
    row = {
        "environment": environment.key,
        "environment_label": environment.label,
        "method": method.key,
        "method_label": method.label,
        "seed": seed,
        "nominal_timesteps": int(budget),
        "actual_timesteps": steps,
        "early_stop_triggered": bool(section.get("early_stop_triggered", False)),
        "run_directory": reference(directory),
        "config_source": reference(directory / "config.json"),
        "summary_source": reference(directory / "summary.json"),
        **{metric: None for metric in METRICS},
    }
    group = "pspo" if method.adaptive else GROUPS[method.key]
    matches = completions.get((seed, group), [])
    row["launcher_records"] = matches
    row["process_status"] = (
        "missing_completion" if not matches else "ambiguous_completions"
    )
    if len(matches) == 1:
        record = matches[0]
        row.update(
            process_source=record["source"],
            launcher_host=record["host"],
            cpu_core=record["core"],
        )
        if group == "baselines_lag":
            row["combined_lag_pid_process_s"] = record["seconds"]
            row["process_status"] = "combined_lag_pid_not_separable"
        else:
            row["rl_process_s"] = record["seconds"]
            row["process_status"] = "complete_rl_process"
    tb_root = directory / "tensorboard"
    if method.key in ("ppo_lagrangian", "ppo_pid_lagrangian", "cpo"):
        tb_root /= method.key
    paths = sorted(tb_root.rglob("events.out.tfevents.*"))
    stream, status, streams = select_event_stream(paths, steps)
    row.update(
        logged_interval_status=status,
        event_stream_count=len(paths),
        event_streams=streams,
    )
    if stream:
        row.update(
            logged_interval_s=stream["seconds"],
            event_source=stream["source"],
            event_host=stream["host"],
            logged_start_wall_time=stream["start_wall_time"],
            logged_end_wall_time=stream["end_wall_time"],
        )
    add_pipeline_timing(row, directory, method, config, summary)
    timing_path = directory / "training_time.json"
    timing = read_json(timing_path)
    timers = section.get("timing", {})
    row["training_loop_s"] = seconds(
        timing.get("rl_training_s", timers.get("training_wall_time_s"))
    )
    row["rl_stage_s"] = seconds(timing.get("rl_stage_wall_time_s"))
    row["finalisation_s"] = seconds(timers.get("finalisation_wall_time_s"))
    if timing:
        row["timing_source"] = reference(timing_path)
    diagnostics = section.get("adaptive_diagnostics", {})
    for field, timer_key, diagnostic_key in (
        ("lid_s", "lid_complete_s", "rashomon_wall_time_total_s"),
        (
            "verification_s",
            "exact_verification_s",
            "exact_verification_wall_time_total_s",
        ),
        ("projection_s", "projection_s", "projection_wall_time_total_s"),
    ):
        row[field] = seconds(timers.get(timer_key, diagnostics.get(diagnostic_key)))
    row["finalisation_status"] = (
        "separately_instrumented_within_process"
        if row["finalisation_s"] is not None
        else "included_in_process_not_separable"
    )
    row["initialisation_shared_across_seeds"] = method.adaptive
    if method.adaptive:
        base = Path(config["base_policy_path"])
        base_summary = base.parent / "summary.json"
        row["initialisation_source"] = reference(base_summary)
        row["initialisation_s"] = seconds(timing.get("policy_initialisation_s"))
        if row["initialisation_s"] is not None:
            row["initialisation_source"] = reference(timing_path)
        if row["initialisation_s"] is None:
            row["initialisation_s"] = seconds(
                read_json(base_summary)
                .get("timing", {})
                .get("policy_initialisation_wall_time_s")
            )
        row["initialisation_status"] = (
            "missing_shared_initialisation_timer"
            if row["initialisation_s"] is None
            else "shared_once_per_environment"
        )
        if row["initialisation_s"] is not None and row["rl_process_s"] is not None:
            row["total_s"] = row["initialisation_s"] + row["rl_process_s"]
            row["total_status"] = "shared_initialisation_plus_process_cold_single_seed"
        else:
            row["total_status"] = "incomplete_stage_coverage"
    else:
        row["initialisation_status"] = "random_actor_setup_included_in_process"
        row["total_s"] = row["rl_process_s"]
        row["total_status"] = (
            "complete_method_process"
            if row["total_s"] is not None
            else "missing_individual_process_timer"
        )
    return row


def aggregate_values(values: list[float]) -> dict:
    count = len(values)
    return {
        "n": count,
        "mean_s": statistics.mean(values) if count else None,
        "two_se_s": 2 * statistics.stdev(values) / math.sqrt(count)
        if count > 1
        else None,
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    result = []
    for environment in ENVIRONMENTS:
        for method in METHODS:
            seeds = [
                row
                for row in rows
                if row["environment"] == environment.key and row["method"] == method.key
            ]
            entry = {
                "environment": environment.key,
                "environment_label": environment.label,
                "method": method.key,
                "method_label": method.label,
                "expected_seeds": len(EXPECTED_SEEDS),
                "actual_timesteps_min": min(row["actual_timesteps"] for row in seeds),
                "actual_timesteps_max": max(row["actual_timesteps"] for row in seeds),
            }
            for metric in METRICS:
                stats = aggregate_values(
                    [row[metric] for row in seeds if row[metric] is not None]
                )
                entry.update({f"{metric}_{key}": value for key, value in stats.items()})
            result.append(entry)
    return result


def cell(row: dict, metric: str, latex: bool = False) -> str:
    count = row[f"{metric}_n"]
    if not count:
        return "---" if latex else "N/A (n=0)"
    mean = row[f"{metric}_mean_s"] / 60
    error = row[f"{metric}_two_se_s"]
    if error is None:
        return f"{mean:.2f} (n=1; SE unavailable)"
    return (
        f"${mean:.2f} \\pm {error / 60:.2f}$ ({count})"
        if latex
        else f"{mean:.2f} ± {error / 60:.2f} (n={count})"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


NOTES = """All durations are wall-clock seconds in the CSVs, minutes in tables. Cells are
mean ± two standard errors (sample SD, ddof=1); n counts available seeds, not
necessarily all ten. A single observation has no estimated SE. N/A is not zero.

Launcher RL process elapsed includes interpreter/model setup, training, periodic
and final evaluation, and saving. It is not a learn()-only timer. Integer launcher
durations have approximately one-second resolution. Instrumented learn() and RL
stage timers, where present, are separate columns, never mixed with launcher times.
Parent pipeline worker-stage timers are also separate: they include stage-module
import/argument parsing and stage execution, but exclude outer pipeline setup and
interpreter startup. A surviving parent record must match the exact selected
stage summary, directory, environment, budget, and (when available) event interval.
Concurrent method groups overwrite the parent summary, so coverage is incomplete.
Finalisation is included in process elapsed but not independently recoverable for
these historical runs. No separate post-training safe-LID projection is performed
in this selected comparison. Common pre-existing environment/shield construction
is outside the method-execution boundary, including for PPO-Shield.

PSPO behaviour-cloning initialisation is shared once per environment; its timer
is unavailable in the selected historical cohorts. PSPO totals therefore remain
N/A. If recorded, total_s is shared initialisation + full process elapsed for a
cold single-seed execution, not an amortised ten-seed cost. Do not sum the shared
initialisation across seeds. In-training LID/verification/projection timings are
nested diagnostics, already within RL execution; they must not be added again.

TensorBoard logged interval is writer-start event wall_time to the last logged
event timestamp, only for a unique stream aligned with actual final training
steps. It may include setup/evaluation and omit final updates/evaluation/saving.
It is neither a full process timer nor a guaranteed learn()-only timer. Failed or
incomplete streams are not merged with retries. Multiple final-step-aligned streams
are ambiguous and remain N/A. No filesystem timestamps are used.

The Lagrangian launcher runs Lag and PID together in one job, with parallel workers.
Its joint elapsed time is preserved in seed data but cannot be attributed to either
method (nor divided equally).
Only explicit successful completion records from selected cohort logs are accepted;
repeated successful completions are ambiguous, never averaged or silently replaced.

Runs used different machines and concurrent loads. Recorded launcher hosts, CPU
core IDs, and TensorBoard hostnames are supplied where available; absence of a host
is not evidence of common hardware. These are historical observed runtimes, not a
hardware-controlled speed benchmark. Nominal budgets and actual rollout-rounded
steps are preserved per seed. No reruns were launched.
"""


def write_reports(output: Path, rows: list[dict], aggregates: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_time_seeds.csv", rows)
    write_csv(output / "training_time_aggregates.csv", aggregates)
    process_count = sum(row["rl_process_s"] is not None for row in rows)
    logged_count = sum(row["logged_interval_s"] is not None for row in rows)
    report = [
        "# Main baseline comparison: recorded training times",
        "",
        f"Extracted {len(rows)} runs (six methods × six environments × ten seeds). "
        f"Full RL-process timings: {process_count}; partial logged intervals: {logged_count}.",
        "",
        "Tables show minutes, mean ± two standard errors, with available-seed counts. "
        "PSPO process times exclude shared initialisation; its full totals are unavailable. "
        "Historical runtimes are not hardware-controlled. See timing boundaries below.",
        "",
    ]
    for metric, title in METRICS.items():
        available = any(row[f"{metric}_n"] for row in aggregates)
        report += [f"## {title} (minutes)", ""]
        if available:
            report += [
                "| Environment | "
                + " | ".join(method.label for method in METHODS)
                + " |",
                "|---|" + "---|" * len(METHODS),
            ]
        else:
            report += [
                "Unavailable as a separately measured component for all 36 method/environment pairs."
            ]
        tex = [
            f"% {title}",
            "% Minutes: mean +/- 2SE; parenthesised count is n. Missing is not zero.",
            "\\begin{tabular}{l" + "r" * len(METHODS) + "}",
            "\\toprule",
            "Environment & " + " & ".join(method.label for method in METHODS) + r" \\",
            "\\midrule",
        ]
        for environment in ENVIRONMENTS:
            selected = [
                next(
                    row
                    for row in aggregates
                    if row["environment"] == environment.key
                    and row["method"] == method.key
                )
                for method in METHODS
            ]
            if available:
                report.append(
                    "| "
                    + environment.label
                    + " | "
                    + " | ".join(cell(row, metric) for row in selected)
                    + " |"
                )
            tex.append(
                environment.label
                + " & "
                + " & ".join(cell(row, metric, True) for row in selected)
                + r" \\"
            )
        tex += ["\\bottomrule", "\\end{tabular}", ""]
        (output / f"{metric}_table.tex").write_text("\n".join(tex))
        report.append("")
    report += [
        "## Timing boundaries and provenance",
        "",
        NOTES.strip(),
        "",
        "Regenerate with:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/pspo-aamas-matplotlib .venv/bin/python "
        "projects/safe_policy_optimisation/scripts/generate_training_time_comparison.py",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(report))
    provenance = {
        "generator": reference(Path(__file__).resolve()),
        "scope": "main baseline comparison",
        "seeds": list(EXPECTED_SEEDS),
        "notes": NOTES,
        "environments": [
            {
                "key": env.key,
                "nominal_budget": env.nominal_budget,
                "baseline_root": reference(env.baseline_root),
                "adaptive_root": reference(env.adaptive_root),
                "baseline_logs": [
                    reference(path) for path in launcher_paths(env, False)
                ],
                "pspo_logs": [reference(path) for path in launcher_paths(env, True)],
            }
            for env in ENVIRONMENTS
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = []
    for environment in ENVIRONMENTS:
        completions = {
            adaptive: read_completions(environment, adaptive)
            for adaptive in (False, True)
        }
        for method in METHODS:
            for seed in EXPECTED_SEEDS:
                rows.append(
                    build_seed_row(
                        environment, method, seed, completions[method.adaptive]
                    )
                )
        print(f"Extracted {environment.label}", flush=True)
    aggregates = aggregate_rows(rows)
    write_reports(args.output_dir, rows, aggregates)
    print(
        f"Wrote {len(rows)} seed rows and {len(aggregates)} aggregates to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
