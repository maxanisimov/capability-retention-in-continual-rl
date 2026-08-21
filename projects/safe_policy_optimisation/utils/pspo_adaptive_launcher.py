"""Validation helpers for PSPO-adaptive experiment launchers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_cpu_ids(value: str) -> list[int]:
    """Parse comma/whitespace separated CPU ids and inclusive ranges."""

    tokens = value.replace(",", " ").split()
    cpu_ids: list[int] = []
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid CPU range {token!r}") from exc
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range {token!r}")
            cpu_ids.extend(range(start, end + 1))
        else:
            try:
                cpu_id = int(token)
            except ValueError as exc:
                raise ValueError(f"invalid CPU id {token!r}") from exc
            if cpu_id < 0:
                raise ValueError("CPU ids must be non-negative")
            cpu_ids.append(cpu_id)
    if not cpu_ids:
        raise ValueError("at least one CPU id is required")
    if len(set(cpu_ids)) != len(cpu_ids):
        raise ValueError("CPU ids must be unique")
    return cpu_ids


def resolve_seed_cpu_ids(
    seeds: list[int],
    *,
    cpu_ids: str = "",
    core_start: str = "",
) -> list[int]:
    """Map one distinct CPU to each seed using an explicit list or a range."""

    if cpu_ids:
        resolved = parse_cpu_ids(cpu_ids)
        if len(resolved) != len(seeds):
            raise ValueError(
                f"CPU_IDS supplies {len(resolved)} CPUs for {len(seeds)} seeds"
            )
        return resolved
    if not core_start:
        raise ValueError("set CPU_IDS or CORE_START")
    try:
        first = int(core_start)
    except ValueError as exc:
        raise ValueError("CORE_START must be a non-negative integer") from exc
    if first < 0:
        raise ValueError("CORE_START must be a non-negative integer")
    return list(range(first, first + len(seeds)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_certificate_samples(
    value: str,
    *,
    default: int,
    shield_mask: np.ndarray | None = None,
) -> int:
    """Resolve an integer certificate size or exhaustive ``all`` coverage."""

    if value == "all":
        if shield_mask is None:
            raise ValueError("shield_mask is required for exhaustive certificate coverage")
        mask = np.asarray(shield_mask)
        if mask.ndim != 2:
            raise ValueError(f"shield_mask must be two-dimensional, got {mask.shape}")
        certificate_samples = int((mask.sum(axis=1) > 0).sum())
    elif value:
        try:
            certificate_samples = int(value)
        except ValueError as exc:
            raise ValueError(
                "certificate samples must be a positive integer or 'all'"
            ) from exc
    else:
        certificate_samples = int(default)

    if certificate_samples <= 0:
        raise ValueError("certificate samples must be positive")
    return certificate_samples


def resolve_target_margin(value: str, *, default: float) -> float:
    """Resolve and validate an optional behaviour-cloning target margin."""

    target_margin = float(value) if value else float(default)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("target margin must be finite and non-negative")
    return target_margin


def base_policy_artifact_matches(
    base_dir: Path,
    *,
    shield_path: Path,
    dataset_size: int,
    hidden_dim: int,
    n_hidden: int,
    state_representation: str,
    margin_mode: str,
    target_margin: float,
) -> bool:
    """Return whether a prepared base policy exactly matches this experiment."""

    required = (
        base_dir / "base_policy.pt",
        base_dir / "safe_behaviour_dataset.pt",
        base_dir / "summary.json",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        summary = json.loads(required[-1].read_text())
        shield_sha256 = _file_sha256(shield_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict) or summary.get("base_policy_only") is not True:
        return False
    architecture = summary.get("architecture")
    dataset = summary.get("dataset")
    base_policy = summary.get("base_policy")
    if not all(isinstance(value, dict) for value in (architecture, dataset, base_policy)):
        return False
    try:
        return bool(
            summary.get("shield_sha256") == shield_sha256
            and int(dataset["dataset_size"]) == int(dataset_size)
            and int(architecture["hidden_dim"]) == int(hidden_dim)
            and int(architecture["n_hidden"]) == int(n_hidden)
            and architecture["state_representation"] == state_representation
            and base_policy["bc_margin_mode"] == margin_mode
            and bool(base_policy["reached_target"])
            and math.isclose(
                float(base_policy["target_margin"]),
                float(target_margin),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def initial_safe_set_matches(
    safe_set_dir: Path,
    *,
    multi_label_mode: str,
    surrogate: str,
    target_margin: float,
    certificate_samples: int,
    n_iters: int,
) -> bool:
    """Return whether an initial safe set matches every reusable setting."""

    required = (
        safe_set_dir / "rashomon_param_bounds.pt",
        safe_set_dir / "base_policy.pt",
        safe_set_dir / "summary.json",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        summary = json.loads(required[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    base_policy = summary.get("base_policy")
    rashomon = summary.get("rashomon")
    if not isinstance(base_policy, dict) or not isinstance(rashomon, dict):
        return False
    try:
        stored_target_margin = float(base_policy["target_margin"])
        stored_certificate_samples = int(rashomon["certificate_samples"])
        stored_n_iters = int(rashomon["n_iters"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        base_policy.get("bc_margin_mode") == multi_label_mode
        and rashomon.get("multi_label_mode") == multi_label_mode
        and rashomon.get("surrogate") == surrogate
        and math.isclose(
            stored_target_margin,
            float(target_margin),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and stored_certificate_samples == int(certificate_samples)
        and stored_n_iters == int(n_iters)
    )
