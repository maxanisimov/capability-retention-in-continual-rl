"""Compute a Rashomon set from a precomputed tabular safety shield.

The saved shield provides a binary safe-action mask indexed by discrete state id.
This script converts that table into the same one-hot state representation used by
the local PPO-Lagrangian baseline, fits a base policy on the resulting safe-action
demonstration dataset, and computes an IBP Rashomon set around that base policy.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from abstract_gradient_training.bounded_models import StateIdLookupLinear
from provably_safe_policy_optimisation.regions import zonotope_rank_default
from torch.utils.data import DataLoader, TensorDataset

from projects.safe_policy_optimisation.utils.io import write_json
from projects.safe_policy_optimisation.utils.log import log_info
from projects.safe_policy_optimisation.utils.pspo_defaults import (
    BC_INITIALISATION_OBJECTIVE,
    BC_MARGIN_MODE,
    BC_MAX_UNSAFE_MASS,
    BC_MIN_SAFE_ACTION_ENTROPY,
    BC_SAFE_ACTION_ENTROPY_WEIGHT,
    BC_SAFE_ACTION_UNIFORMITY_WEIGHT,
    BC_TARGET_MARGIN,
    BC_UNSAFE_MASS_TARGET,
    RASHOMON_MULTI_LABEL_MODE,
    RASHOMON_SURROGATE,
)
from projects.safe_policy_optimisation.utils.rashomon import (
    parse_rashomon_batch_size,
    resolve_rashomon_batch_size,
)
from projects.safe_policy_optimisation.utils.shield import (
    load_shield_mask as _load_shield_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "shield_rashomon"
)


def parse_nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1]")
    return parsed


def parse_unsafe_mass_target(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 0.5:
        raise argparse.ArgumentTypeError("value must lie strictly between 0 and 0.5")
    return parsed


def parse_max_unsafe_mass(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 0.5:
        raise argparse.ArgumentTypeError("value must lie in [0, 0.5)")
    return parsed


def load_shield_mask(shield_path: Path, *, risk_threshold: float | None = None) -> np.ndarray:
    """Load a float32 ``(state, action)`` shield mask, auto-detecting the source.

    Rashomon-set construction needs the mask as float32 safe-action indicators,
    so this delegates to :func:`...utils.shield.load_shield_mask` with
    ``source="auto"``.
    """

    return _load_shield_mask(
        shield_path, source="auto", risk_threshold=risk_threshold, dtype=np.float32
    )


def make_safe_behaviour_payload(
    mask: np.ndarray,
    state_to_features: Any = None,
    *,
    state_representation: str = "one_hot",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """State features and multi-hot safe actions for the BC dataset.

    ``state_to_features`` (the env's ``state_to_features``) selects the state
    representation the base policy -- and hence the PSPO actor -- is fitted on:

    * ``None`` with ``state_representation="one_hot"``: one-hot over discrete
      state IDs (the historical view).
    * ``None`` with ``state_representation="state_id_lookup"``: an int64 state
      ID whose lookup layer is exactly equivalent to the one-hot linear layer.
    * a callable: the env's normalised decoded features, so the base policy has
      the same feature input the deployed PSPO actor receives.
    """

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected shield mask shape (n_states, n_actions), got {mask.shape}.")
    n_states, n_actions = mask.shape
    safe_counts = mask.sum(axis=1)
    safe_state_ids = np.flatnonzero(safe_counts > 0)
    if safe_state_ids.size == 0:
        raise ValueError("Shield contains no states with at least one safe action.")

    if state_representation not in {"one_hot", "features", "state_id_lookup"}:
        raise ValueError(
            "state_representation must be 'one_hot', 'features', or "
            f"'state_id_lookup'; got {state_representation!r}."
        )
    if state_representation == "state_id_lookup" and state_to_features is not None:
        raise ValueError(
            "state_id_lookup and state_to_features are mutually exclusive."
        )
    if state_to_features is not None:
        feature_rows = np.stack([np.asarray(state_to_features(int(s))) for s in safe_state_ids])
        states = torch.as_tensor(feature_rows, dtype=torch.float32)
        representation = "decoded_features"
        input_dim = int(states.shape[1])
    elif state_representation == "state_id_lookup":
        states = torch.as_tensor(safe_state_ids, dtype=torch.long).reshape(-1, 1)
        representation = "state_id_lookup_discrete_observation"
        input_dim = int(n_states)
    else:
        states = torch.nn.functional.one_hot(
            torch.as_tensor(safe_state_ids, dtype=torch.long),
            num_classes=int(n_states),
        ).to(torch.float32)
        representation = "one_hot_discrete_observation"
        input_dim = int(n_states)
    actions = torch.as_tensor(mask[safe_state_ids], dtype=torch.float32)
    metadata = {
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "feature_dim": input_dim,
        "input_dim": input_dim,
        "stored_state_shape": [int(value) for value in states.shape],
        "stored_state_dtype": str(states.dtype),
        "stored_state_bytes": int(states.numel() * states.element_size()),
        "dataset_size": int(safe_state_ids.size),
        "excluded_no_safe_action_states": int(n_states - safe_state_ids.size),
        "state_representation": representation,
        "safe_state_ids": [int(x) for x in safe_state_ids.tolist()],
    }
    return {"state": states, "actions": actions}, metadata


def build_base_policy(
    input_dim: int,
    n_actions: int,
    *,
    hidden_dim: int,
    n_hidden: int,
    state_id_lookup: bool = False,
) -> nn.Sequential:
    """Build a Sequential policy compatible with IntervalTrainer."""

    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_index in range(int(n_hidden)):
        if hidden_index == 0 and state_id_lookup:
            layers.append(StateIdLookupLinear(last_dim, int(hidden_dim)))
        else:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        last_dim = int(hidden_dim)
    if int(n_hidden) == 0 and state_id_lookup:
        layers.append(StateIdLookupLinear(last_dim, int(n_actions)))
    else:
        layers.append(nn.Linear(last_dim, int(n_actions)))
    return nn.Sequential(*layers)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_policy_for_dataset(
    base_policy_path: Path,
    dataset: dict[str, torch.Tensor],
    dataset_metadata: dict[str, Any],
    *,
    hidden_dim: int,
    n_hidden: int,
    target_margin: float,
    margin_mode: str,
    device: str | torch.device,
    margin_loss_weight: float = 1.0,
    safe_action_entropy_weight: float = 0.0,
    min_safe_action_entropy: float = 0.95,
    initialisation_objective: str = "margin",
    unsafe_mass_target: float = 0.01,
    max_unsafe_mass: float = 0.02,
    safe_action_uniformity_weight: float = 1.0,
) -> tuple[nn.Sequential, dict[str, Any], dict[str, Any]]:
    """Load and validate an existing BC policy for Rashomon-set growth.

    This is used by compute-matched PSPO comparisons, where refitting the BC
    policy would change the centre of the safe parameter region and confound
    the comparison. The returned metrics are recomputed on the current shield
    dataset rather than trusted from the saved payload.
    """

    _validate_base_policy_initialisation_settings(
        initialisation_objective=initialisation_objective,
        margin_loss_weight=margin_loss_weight,
        safe_action_entropy_weight=safe_action_entropy_weight,
        min_safe_action_entropy=min_safe_action_entropy,
        unsafe_mass_target=unsafe_mass_target,
        max_unsafe_mass=max_unsafe_mass,
        safe_action_uniformity_weight=safe_action_uniformity_weight,
    )

    base_policy_path = Path(base_policy_path).resolve()
    payload = torch.load(base_policy_path, map_location="cpu", weights_only=False)
    for key in ("architecture", "state_dict"):
        if key not in payload:
            raise KeyError(
                f"Base policy file must contain {key!r}; keys={sorted(payload.keys())}."
            )

    architecture = dict(payload["architecture"])
    expected = {
        "input_dim": int(dataset_metadata.get("input_dim", dataset["state"].shape[1])),
        "n_actions": int(dataset["actions"].shape[1]),
        "hidden_dim": int(hidden_dim),
        "n_hidden": int(n_hidden),
        "activation": "Tanh",
        "state_representation": str(dataset_metadata["state_representation"]),
    }
    mismatches = {
        key: {"expected": value, "actual": architecture.get(key)}
        for key, value in expected.items()
        if architecture.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Existing base policy is incompatible with the requested safe-set "
            f"configuration: {mismatches}."
        )

    model = build_base_policy(
        expected["input_dim"],
        expected["n_actions"],
        hidden_dim=expected["hidden_dim"],
        n_hidden=expected["n_hidden"],
        state_id_lookup=(
            expected["state_representation"]
            == "state_id_lookup_discrete_observation"
        ),
    )
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    model.to(torch.device(device))

    saved_metrics = payload.get("bc_metrics")
    stored_objective = (
        str(saved_metrics.get("initialisation_objective", "margin"))
        if isinstance(saved_metrics, dict)
        else "margin"
    )
    settings_match = stored_objective == initialisation_objective
    if initialisation_objective == "safe_mass":
        if not isinstance(saved_metrics, dict):
            raise ValueError(
                "Safe-mass initialisation cannot reuse a legacy base policy "
                "without safe-mass objective metadata."
            )
        for key, requested in (
            ("unsafe_mass_target", unsafe_mass_target),
            ("max_unsafe_mass", max_unsafe_mass),
            ("safe_action_uniformity_weight", safe_action_uniformity_weight),
            ("min_safe_action_entropy", min_safe_action_entropy),
        ):
            try:
                settings_match = settings_match and math.isclose(
                    float(saved_metrics[key]),
                    float(requested),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (KeyError, TypeError, ValueError):
                settings_match = False
    else:
        stored_margin_loss_weight = (
            float(saved_metrics.get("margin_loss_weight", 1.0))
            if isinstance(saved_metrics, dict)
            else 1.0
        )
        settings_match = settings_match and math.isclose(
            stored_margin_loss_weight,
            float(margin_loss_weight),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        requested_entropy_weight = float(safe_action_entropy_weight)
        stored_entropy_weight = (
            float(saved_metrics.get("safe_action_entropy_weight", 0.0))
            if isinstance(saved_metrics, dict)
            else 0.0
        )
        settings_match = settings_match and math.isclose(
            stored_entropy_weight,
            requested_entropy_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    if initialisation_objective == "margin" and safe_action_entropy_weight > 0.0:
        if not isinstance(saved_metrics, dict):
            raise ValueError(
                "Entropy-enabled initialisation cannot reuse a legacy base policy "
                "without BC entropy metadata."
            )
        try:
            settings_match = settings_match and math.isclose(
                float(saved_metrics["min_safe_action_entropy"]),
                float(min_safe_action_entropy),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (KeyError, TypeError, ValueError):
            settings_match = False
    if not settings_match:
        raise ValueError(
            "Existing base policy was not fitted with the requested "
            "initialisation objective and settings."
        )

    final_accuracy = allowed_action_accuracy(model, dataset, device=device)
    final_any_margin = minimum_safe_action_margin(model, dataset, device=device, mode="any")
    final_all_margin = minimum_safe_action_margin(model, dataset, device=device, mode="all")
    final_margin = final_any_margin if margin_mode == "any" else final_all_margin
    diagnostics = safe_action_initialisation_diagnostics(model, dataset, device=device)
    entropy_enabled = (
        float(safe_action_uniformity_weight) > 0.0
        if initialisation_objective == "safe_mass"
        else float(safe_action_entropy_weight) > 0.0
    )
    entropy_reached = (
        not entropy_enabled
        or diagnostics["normalized_safe_action_entropy_min"]
        >= float(min_safe_action_entropy)
    )
    if initialisation_objective == "safe_mass":
        reached_target = bool(
            final_accuracy >= 1.0
            and diagnostics["max_unsafe_action_mass"] <= float(max_unsafe_mass)
            and entropy_reached
        )
    else:
        reached_target = bool(
            final_accuracy >= 1.0
            and (
                final_margin > 0.0
                if float(margin_loss_weight) == 0.0
                else final_margin >= float(target_margin)
            )
            and entropy_reached
        )
    bc_metrics = {
        "initial_accuracy": final_accuracy,
        "final_accuracy": final_accuracy,
        "epochs_run": 0,
        "reached_target": reached_target,
        "used_direct_linear_init": False,
        "initial_min_margin": final_margin,
        "final_min_margin": final_margin,
        "initial_min_any_margin": final_any_margin,
        "final_min_any_margin": final_any_margin,
        "initial_min_all_margin": final_all_margin,
        "final_min_all_margin": final_all_margin,
        "target_margin": float(target_margin),
        "margin_loss_weight": float(margin_loss_weight),
        "stopping_criterion": (
            _feasibility_criterion_name(margin_mode)
            if initialisation_objective == "margin" and float(margin_loss_weight) == 0.0
            else "target_margin"
        ),
        "bc_margin_mode": margin_mode,
        "initialisation_objective": initialisation_objective,
        "unsafe_mass_target": float(unsafe_mass_target),
        "max_unsafe_mass": float(max_unsafe_mass),
        "safe_action_uniformity_weight": float(safe_action_uniformity_weight),
        "safe_action_entropy_weight": float(safe_action_entropy_weight),
        "min_safe_action_entropy": float(min_safe_action_entropy),
        **{f"initial_{key}": value for key, value in diagnostics.items()},
        **{f"final_{key}": value for key, value in diagnostics.items()},
        "source": "loaded_existing_policy",
    }
    source_metadata = {
        "path": str(base_policy_path),
        "sha256": _file_sha256(base_policy_path),
    }
    return model, bc_metrics, source_metadata


def initialise_linear_policy_from_masks(model: nn.Sequential, dataset: dict[str, torch.Tensor], *, margin: float) -> bool:
    """Closed-form BC initializer for one-hot or lookup tabular policies."""

    if len(model) != 1 or not isinstance(model[0], (nn.Linear, StateIdLookupLinear)):
        return False
    states = dataset["state"]
    actions = dataset["actions"]
    state_ids = (
        states.reshape(-1).long()
        if isinstance(model[0], StateIdLookupLinear)
        else states.argmax(dim=1)
    )
    with torch.no_grad():
        layer = model[0]
        layer.weight.zero_()
        layer.bias.zero_()
        actions = actions.to(layer.weight.device)
        for row_idx, state_id in enumerate(state_ids.tolist()):
            safe = actions[row_idx].bool()
            layer.weight[:, int(state_id)] = torch.where(
                safe,
                torch.full_like(layer.weight[:, int(state_id)], float(margin)),
                torch.full_like(layer.weight[:, int(state_id)], -float(margin)),
            )
    return True


@torch.no_grad()
def allowed_action_accuracy(model: nn.Module, dataset: dict[str, torch.Tensor], *, device: str | torch.device) -> float:
    """Fraction of rows where the greedy action is inside the multi-hot safe set."""

    device_t = torch.device(device)
    model.eval()
    states = dataset["state"].to(device_t)
    actions = dataset["actions"].to(device_t)
    logits = model(states)
    preds = logits.argmax(dim=1)
    correct = actions[torch.arange(actions.shape[0], device=device_t), preds] > 0
    return float(correct.float().mean().item())


def safe_action_bc_loss(logits: torch.Tensor, safe_actions: torch.Tensor) -> torch.Tensor:
    """Negative log probability assigned to the set of safe actions."""

    allowed = safe_actions.bool()
    masked_logits = logits.masked_fill(~allowed, -1e9)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(masked_logits, dim=1)).mean()


def safe_unsafe_mass_cross_entropy_loss(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
    *,
    unsafe_mass_target: float,
) -> torch.Tensor:
    """Cross entropy for aggregate safe/unsafe probability mass.

    The target distribution is ``(1 - epsilon, epsilon)`` over the safe and
    unsafe action sets. A small non-zero ``epsilon`` gives this objective a
    finite optimum, unlike driving unsafe mass exactly to zero. Rows where all
    actions are safe have no unsafe category and are excluded from this term.
    """

    if logits.shape != safe_actions.shape:
        raise ValueError(
            "Logits and safe-action masks must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(safe_actions.shape)}."
        )
    if not math.isfinite(unsafe_mass_target) or not 0.0 < unsafe_mass_target < 0.5:
        raise ValueError("unsafe_mass_target must lie strictly between 0 and 0.5.")
    allowed = safe_actions.bool()
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("Every row must contain at least one safe action.")
    contested = (~allowed).any(dim=1)
    if not bool(contested.any()):
        return logits.sum() * 0.0

    log_normalizer = torch.logsumexp(logits[contested], dim=1)
    contested_logits = logits[contested]
    contested_allowed = allowed[contested]
    log_safe_mass = (
        torch.logsumexp(
            contested_logits.masked_fill(~contested_allowed, float("-inf")), dim=1
        )
        - log_normalizer
    )
    log_unsafe_mass = (
        torch.logsumexp(
            contested_logits.masked_fill(contested_allowed, float("-inf")), dim=1
        )
        - log_normalizer
    )
    epsilon = float(unsafe_mass_target)
    return (-(1.0 - epsilon) * log_safe_mass - epsilon * log_unsafe_mass).mean()


def safe_action_uniformity_loss(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
) -> torch.Tensor:
    """Mean ``KL(U(safe_actions) || policy(. | safe_actions))``.

    The loss is zero exactly when probability is uniform among a state's safe
    actions. Unsafe logits do not affect it, keeping safe-action diversity and
    aggregate safety mass as separate, interpretable objectives.
    """

    if logits.shape != safe_actions.shape:
        raise ValueError(
            "Logits and safe-action masks must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(safe_actions.shape)}."
        )
    allowed = safe_actions.bool()
    safe_counts = allowed.sum(dim=1)
    if not bool((safe_counts > 0).all()):
        raise ValueError("Every row must contain at least one safe action.")
    safe_log_normalizer = torch.logsumexp(
        logits.masked_fill(~allowed, float("-inf")), dim=1, keepdim=True
    )
    conditional_log_probs = (logits - safe_log_normalizer).masked_fill(~allowed, 0.0)
    uniform_cross_entropy = -conditional_log_probs.sum(dim=1) / safe_counts.to(logits.dtype)
    return (uniform_cross_entropy - safe_counts.to(logits.dtype).log()).mean()


def safe_action_conditional_probabilities(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
) -> torch.Tensor:
    """Policy probabilities renormalised over each state's safe actions only."""

    if logits.shape != safe_actions.shape:
        raise ValueError(
            "Logits and safe-action masks must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(safe_actions.shape)}."
        )
    allowed = safe_actions.bool()
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("Every row must contain at least one safe action.")
    return torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=1)


def normalized_safe_action_entropies(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state entropy conditioned on safety, normalised to the interval [0, 1].

    The returned mask identifies states with more than one safe action. A state
    with exactly one safe action has no safe choice to preserve and is assigned
    entropy 1 by convention, while being excluded from the entropy loss.
    """

    allowed = safe_actions.bool()
    safe_counts = allowed.sum(dim=1)
    conditional_probs = safe_action_conditional_probabilities(logits, safe_actions)
    entropy = -(
        conditional_probs
        * conditional_probs.clamp_min(torch.finfo(logits.dtype).tiny).log()
    ).sum(dim=1)
    multi_safe = safe_counts > 1
    normalized = torch.ones_like(entropy)
    normalized[multi_safe] = (
        entropy[multi_safe]
        / safe_counts[multi_safe].to(logits.dtype).log()
    ).clamp(0.0, 1.0)
    return normalized, multi_safe


def safe_action_entropy_loss(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
) -> torch.Tensor:
    """Penalise concentration within the policy distribution over safe actions."""

    normalized_entropy, multi_safe = normalized_safe_action_entropies(
        logits, safe_actions
    )
    if not bool(multi_safe.any()):
        return logits.new_zeros(())
    return (1.0 - normalized_entropy[multi_safe]).mean()


@torch.no_grad()
def safe_action_initialisation_diagnostics(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Reward-free diagnostics for the initial policy's safe-action geometry."""

    device_t = torch.device(device)
    model.eval()
    logits = model(dataset["state"].to(device_t))
    safe_actions = dataset["actions"].to(device_t)
    allowed = safe_actions.bool()
    normalized_entropy, multi_safe = normalized_safe_action_entropies(
        logits, safe_actions
    )
    entropy_values = normalized_entropy[multi_safe]
    if not bool(multi_safe.any()):
        entropy_values = normalized_entropy.new_ones(1)
    unsafe_mass = (torch.softmax(logits, dim=1) * (~allowed)).sum(dim=1)
    return {
        "normalized_safe_action_entropy_mean": float(entropy_values.mean().item()),
        "normalized_safe_action_entropy_p5": float(
            torch.quantile(entropy_values, 0.05).item()
        ),
        "normalized_safe_action_entropy_min": float(entropy_values.min().item()),
        "multi_safe_action_states": int(multi_safe.sum().item()),
        "mean_unsafe_action_mass": float(unsafe_mass.mean().item()),
        "max_unsafe_action_mass": float(unsafe_mass.max().item()),
    }


def safe_action_margins(
    logits: torch.Tensor, safe_actions: torch.Tensor, *, mode: str = "any"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state safe-vs-unsafe logit margin.

    ``mode="any"`` is the historical criterion:
    ``best safe logit - best unsafe logit``. ``mode="all"`` is stricter:
    ``worst safe logit - best unsafe logit``. Returns the margins plus a mask of
    rows that have an unsafe action to compete with; rows where every action is
    safe are unconstrained and excluded.
    """

    if mode not in {"any", "all"}:
        raise ValueError(f"Unknown BC margin mode: {mode!r}. Expected 'any' or 'all'.")
    allowed = safe_actions.bool()
    neg = torch.finfo(logits.dtype).min
    pos = torch.finfo(logits.dtype).max
    if mode == "any":
        safe_logit = logits.masked_fill(~allowed, neg).max(dim=1).values
    else:
        safe_logit = logits.masked_fill(~allowed, pos).min(dim=1).values
    best_unsafe = logits.masked_fill(allowed, neg).max(dim=1).values
    contested = (~allowed).any(dim=1)
    return safe_logit - best_unsafe, contested


def _feasibility_criterion_name(margin_mode: str) -> str:
    """Name of the CE-only (no margin loss) stopping criterion for this mode."""

    return (
        "strict_all_safe_feasibility" if margin_mode == "all" else "any_safe_feasibility"
    )


def safe_action_margin_loss(
    logits: torch.Tensor, safe_actions: torch.Tensor, *, target_margin: float, mode: str = "any"
) -> torch.Tensor:
    """Hinge pushing every contested state to at least ``target_margin``."""

    margins, contested = safe_action_margins(logits, safe_actions, mode=mode)
    if not bool(contested.any()):
        return logits.new_zeros(())
    return torch.relu(float(target_margin) - margins[contested]).mean()


def safe_action_logit_interval_analysis_from_bounds(
    logits_l: torch.Tensor,
    logits_u: torch.Tensor,
    safe_actions: torch.Tensor,
) -> dict[str, Any]:
    """Summarise safe-action preservation from output logit intervals.

    Two related but different quantities are reported:

    * ``safe_vs_unsafe_*``: every safe action whose lower logit bound is above
      every unsafe action's upper logit bound. This is the sound surrogate
      guarantee used by ``multi_label_mode="all"``. States with no unsafe
      action make this condition vacuously true for every safe action.
    * ``possible_argmax_*``: every safe action whose upper logit bound is not
      below another action's lower bound, meaning interval bounds do not rule
      out that action being greedy for some member of the Rashomon set.
    """

    if logits_l.shape != logits_u.shape:
        raise ValueError("Logit lower/upper bounds must have the same shape.")
    if logits_l.shape != safe_actions.shape:
        raise ValueError(
            f"Logit bounds and safe action mask must have the same shape, got "
            f"{tuple(logits_l.shape)} and {tuple(safe_actions.shape)}."
        )

    safe_mask = safe_actions.bool()
    safe_counts = safe_mask.sum(dim=1)
    valid_rows = safe_counts > 0
    if not bool(valid_rows.all().item()):
        raise ValueError("Safe-action analysis expects every row to have at least one safe action.")
    n_rows, n_actions = safe_mask.shape

    safe_vs_unsafe = torch.zeros_like(safe_mask)
    possible_argmax = torch.zeros_like(safe_mask)
    for action_idx in range(int(n_actions)):
        other_actions = torch.ones(n_actions, dtype=torch.bool, device=safe_mask.device)
        other_actions[action_idx] = False

        unsafe_mask = ~safe_mask
        unsafe_mask[:, action_idx] = False
        has_unsafe = unsafe_mask.any(dim=1)
        worst_unsafe_upper = logits_u.masked_fill(~unsafe_mask, float("-inf")).max(dim=1).values
        safe_vs_unsafe[:, action_idx] = safe_mask[:, action_idx] & (
            ~has_unsafe | (logits_l[:, action_idx] >= worst_unsafe_upper)
        )

        best_other_lower = logits_l[:, other_actions].max(dim=1).values
        possible_argmax[:, action_idx] = safe_mask[:, action_idx] & (
            logits_u[:, action_idx] >= best_other_lower
        )

    total_safe_actions = int(safe_counts.sum().item())
    safe_vs_unsafe_counts = safe_vs_unsafe.sum(dim=1)
    possible_argmax_counts = possible_argmax.sum(dim=1)
    safe_vs_unsafe_pct = safe_vs_unsafe_counts.float() / safe_counts.float() * 100.0
    possible_argmax_pct = possible_argmax_counts.float() / safe_counts.float() * 100.0

    breakdown: dict[str, Any] = {}
    unique_counts, count_frequencies = torch.unique(safe_counts, return_counts=True)
    for safe_count, frequency in zip(unique_counts.tolist(), count_frequencies.tolist()):
        mask = safe_counts == int(safe_count)
        breakdown[str(int(safe_count))] = {
            "states": int(frequency),
            "safe_actions": int(safe_mask[mask].sum().item()),
            "safe_vs_unsafe_count": int(safe_vs_unsafe[mask].sum().item()),
            "safe_vs_unsafe_per_state_mean_pct": float(safe_vs_unsafe_pct[mask].mean().item()),
            "possible_argmax_count": int(possible_argmax[mask].sum().item()),
            "possible_argmax_per_state_mean_pct": float(possible_argmax_pct[mask].mean().item()),
        }

    return {
        "states_with_safe_actions": int(n_rows),
        "n_actions": int(n_actions),
        "total_safe_state_actions": total_safe_actions,
        "safe_action_count_distribution": {
            str(int(k)): int(v) for k, v in zip(unique_counts.tolist(), count_frequencies.tolist())
        },
        "safe_vs_unsafe_count": int(safe_vs_unsafe.sum().item()),
        "safe_vs_unsafe_micro_pct": float(safe_vs_unsafe.sum().float().item() / total_safe_actions * 100.0),
        "safe_vs_unsafe_per_state_mean_pct": float(safe_vs_unsafe_pct.mean().item()),
        "safe_vs_unsafe_per_state_min_pct": float(safe_vs_unsafe_pct.min().item()),
        "safe_vs_unsafe_per_state_max_pct": float(safe_vs_unsafe_pct.max().item()),
        "possible_argmax_count": int(possible_argmax.sum().item()),
        "possible_argmax_micro_pct": float(possible_argmax.sum().float().item() / total_safe_actions * 100.0),
        "possible_argmax_per_state_mean_pct": float(possible_argmax_pct.mean().item()),
        "possible_argmax_per_state_min_pct": float(possible_argmax_pct.min().item()),
        "possible_argmax_per_state_max_pct": float(possible_argmax_pct.max().item()),
        "breakdown_by_safe_action_count": breakdown,
    }


@torch.no_grad()
def safe_action_logit_interval_analysis(
    bounded_model: Any,
    dataset: dict[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run ``bound_forward`` and summarise safe-action logit interval metrics."""

    device_t = torch.device(device)
    states = dataset["state"].to(device_t)
    safe_actions = dataset["actions"].to(device_t)
    logits_l, logits_u = bounded_model.bound_forward(states, states)
    return safe_action_logit_interval_analysis_from_bounds(
        logits_l.detach().cpu(),
        logits_u.detach().cpu(),
        safe_actions.detach().cpu(),
    )


@torch.no_grad()
def minimum_safe_action_margin(
    model: nn.Module, dataset: dict[str, torch.Tensor], *, device: str | torch.device, mode: str = "any"
) -> float:
    """Worst-case per-state safety margin over the whole dataset."""

    device_t = torch.device(device)
    model.eval()
    logits = model(dataset["state"].to(device_t))
    margins, contested = safe_action_margins(logits, dataset["actions"].to(device_t), mode=mode)
    if not bool(contested.any()):
        return float("inf")
    return float(margins[contested].min().item())


def _validate_base_policy_initialisation_settings(
    *,
    initialisation_objective: str,
    margin_loss_weight: float = 1.0,
    safe_action_entropy_weight: float,
    min_safe_action_entropy: float,
    unsafe_mass_target: float,
    max_unsafe_mass: float,
    safe_action_uniformity_weight: float,
) -> None:
    if initialisation_objective not in {"margin", "safe_mass"}:
        raise ValueError(
            "initialisation_objective must be 'margin' or 'safe_mass', got "
            f"{initialisation_objective!r}."
        )
    if not np.isfinite(margin_loss_weight) or margin_loss_weight < 0.0:
        raise ValueError("margin_loss_weight must be finite and non-negative.")
    if not np.isfinite(safe_action_entropy_weight) or safe_action_entropy_weight < 0.0:
        raise ValueError("safe_action_entropy_weight must be finite and non-negative.")
    if not np.isfinite(min_safe_action_entropy) or not 0.0 <= min_safe_action_entropy <= 1.0:
        raise ValueError("min_safe_action_entropy must lie in [0, 1].")
    if not np.isfinite(unsafe_mass_target) or not 0.0 < unsafe_mass_target < 0.5:
        raise ValueError("unsafe_mass_target must lie strictly between 0 and 0.5.")
    if not np.isfinite(max_unsafe_mass) or not 0.0 <= max_unsafe_mass < 0.5:
        raise ValueError("max_unsafe_mass must lie in [0, 0.5).")
    if unsafe_mass_target > max_unsafe_mass:
        raise ValueError("unsafe_mass_target must not exceed max_unsafe_mass.")
    if (
        not np.isfinite(safe_action_uniformity_weight)
        or safe_action_uniformity_weight < 0.0
    ):
        raise ValueError(
            "safe_action_uniformity_weight must be finite and non-negative."
        )


def fit_base_policy(
    model: nn.Sequential,
    dataset: dict[str, torch.Tensor],
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    direct_linear_init: bool = True,
    linear_init_margin: float = 10.0,
    target_margin: float = 10.0,
    margin_loss_weight: float = 1.0,
    margin_mode: str = "any",
    safe_action_entropy_weight: float = 0.0,
    min_safe_action_entropy: float = 0.95,
    initialisation_objective: str = "margin",
    unsafe_mass_target: float = 0.01,
    max_unsafe_mass: float = 0.02,
    safe_action_uniformity_weight: float = 1.0,
) -> dict[str, Any]:
    """Fit a base policy using either margin or probability-mass targets.

    ``safe_mass`` replaces the logit-margin target with two independent terms:
    aggregate safe/unsafe mass cross entropy toward ``(1-epsilon, epsilon)``,
    and uniform-target cross entropy within the safe actions. Its stopping rule
    uses maximum unsafe probability mass and minimum safe-action entropy; the
    logit margin is retained only as a diagnostic.
    """

    torch.manual_seed(int(seed))
    if margin_mode not in {"any", "all"}:
        raise ValueError(f"Unknown BC margin mode: {margin_mode!r}. Expected 'any' or 'all'.")
    _validate_base_policy_initialisation_settings(
        initialisation_objective=initialisation_objective,
        margin_loss_weight=margin_loss_weight,
        safe_action_entropy_weight=safe_action_entropy_weight,
        min_safe_action_entropy=min_safe_action_entropy,
        unsafe_mass_target=unsafe_mass_target,
        max_unsafe_mass=max_unsafe_mass,
        safe_action_uniformity_weight=safe_action_uniformity_weight,
    )
    device_t = torch.device(device)
    model.to(device_t)
    used_direct_init = False
    if direct_linear_init and initialisation_objective == "margin":
        used_direct_init = initialise_linear_policy_from_masks(
            model,
            dataset,
            margin=linear_init_margin,
        )

    initial_accuracy = allowed_action_accuracy(model, dataset, device=device_t)
    initial_any_margin = minimum_safe_action_margin(model, dataset, device=device_t, mode="any")
    initial_all_margin = minimum_safe_action_margin(model, dataset, device=device_t, mode="all")
    initial_margin = initial_any_margin if margin_mode == "any" else initial_all_margin
    initial_diagnostics = safe_action_initialisation_diagnostics(
        model, dataset, device=device_t
    )
    final_diagnostics = initial_diagnostics
    final_any_margin = initial_any_margin
    final_all_margin = initial_all_margin
    final_margin = initial_margin
    entropy_enabled = (
        float(safe_action_uniformity_weight) > 0.0
        if initialisation_objective == "safe_mass"
        else float(safe_action_entropy_weight) > 0.0
    )

    def reached_target(
        accuracy: float,
        margin: float,
        all_margin: float,
        diagnostics: dict[str, Any],
    ) -> bool:
        entropy_reached = (
            not entropy_enabled
            or diagnostics["normalized_safe_action_entropy_min"]
            >= float(min_safe_action_entropy)
        )
        if initialisation_objective == "safe_mass":
            return bool(
                accuracy >= 1.0
                and diagnostics["max_unsafe_action_mass"] <= float(max_unsafe_mass)
                and entropy_reached
            )
        margin_reached = (
            margin > 0.0
            if float(margin_loss_weight) == 0.0
            else margin >= float(target_margin)
        )
        return bool(accuracy >= 1.0 and margin_reached and entropy_reached)

    def metrics(
        *,
        final_accuracy: float,
        epochs_run: int,
        final_margin: float,
        final_any_margin: float,
        final_all_margin: float,
        final_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "epochs_run": int(epochs_run),
            "reached_target": reached_target(
                final_accuracy, final_margin, final_all_margin, final_diagnostics
            ),
            "used_direct_linear_init": used_direct_init,
            "initial_min_margin": initial_margin,
            "final_min_margin": final_margin,
            "initial_min_any_margin": initial_any_margin,
            "final_min_any_margin": final_any_margin,
            "initial_min_all_margin": initial_all_margin,
            "final_min_all_margin": final_all_margin,
            "target_margin": float(target_margin),
            "margin_loss_weight": float(margin_loss_weight),
            "stopping_criterion": (
                _feasibility_criterion_name(margin_mode)
                if initialisation_objective == "margin" and float(margin_loss_weight) == 0.0
                else "target_margin"
            ),
            "bc_margin_mode": margin_mode,
            "initialisation_objective": initialisation_objective,
            "unsafe_mass_target": float(unsafe_mass_target),
            "max_unsafe_mass": float(max_unsafe_mass),
            "safe_action_uniformity_weight": float(safe_action_uniformity_weight),
            "safe_action_entropy_weight": float(safe_action_entropy_weight),
            "min_safe_action_entropy": float(min_safe_action_entropy),
            **{f"initial_{key}": value for key, value in initial_diagnostics.items()},
            **{f"final_{key}": value for key, value in final_diagnostics.items()},
        }

    if reached_target(
        initial_accuracy, initial_margin, initial_all_margin, initial_diagnostics
    ):
        return metrics(
            final_accuracy=initial_accuracy,
            epochs_run=0,
            final_margin=initial_margin,
            final_any_margin=initial_any_margin,
            final_all_margin=initial_all_margin,
            final_diagnostics=initial_diagnostics,
        )

    tensor_dataset = TensorDataset(dataset["state"], dataset["actions"])
    loader = DataLoader(
        tensor_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    final_accuracy = initial_accuracy
    epochs_run = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for states, safe_actions in loader:
            states = states.to(device_t)
            safe_actions = safe_actions.to(device_t)
            logits = model(states)
            if initialisation_objective == "safe_mass":
                loss = safe_unsafe_mass_cross_entropy_loss(
                    logits,
                    safe_actions,
                    unsafe_mass_target=unsafe_mass_target,
                )
                if entropy_enabled:
                    loss = loss + float(
                        safe_action_uniformity_weight
                    ) * safe_action_uniformity_loss(logits, safe_actions)
            else:
                loss = safe_action_bc_loss(logits, safe_actions)
                if float(margin_loss_weight) > 0.0:
                    loss = loss + float(margin_loss_weight) * safe_action_margin_loss(
                        logits, safe_actions, target_margin=target_margin, mode=margin_mode
                    )
                if entropy_enabled:
                    loss = loss + float(safe_action_entropy_weight) * safe_action_entropy_loss(
                        logits, safe_actions
                    )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        final_accuracy = allowed_action_accuracy(model, dataset, device=device_t)
        final_any_margin = minimum_safe_action_margin(model, dataset, device=device_t, mode="any")
        final_all_margin = minimum_safe_action_margin(model, dataset, device=device_t, mode="all")
        final_margin = final_any_margin if margin_mode == "any" else final_all_margin
        final_diagnostics = safe_action_initialisation_diagnostics(
            model, dataset, device=device_t
        )
        if reached_target(
            final_accuracy, final_margin, final_all_margin, final_diagnostics
        ):
            break

    return metrics(
        final_accuracy=final_accuracy,
        epochs_run=epochs_run,
        final_margin=final_margin,
        final_any_margin=final_any_margin,
        final_all_margin=final_all_margin,
        final_diagnostics=final_diagnostics,
    )


def calibrate_inverse_temperature(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    *,
    inverse_temp_start: int,
    inverse_temp_max: int,
    device: str | torch.device,
    multi_label_mode: str = "any",
    surrogate: str = "auto",
) -> tuple[int, float, float]:
    """Find the first inverse temperature whose per-state surrogate is feasible.

    For the probability-based ``"any"`` surrogate, the returned value is the
    minimum valid-action mass and the returned threshold is that same state's
    cardinality-specific threshold.
    """

    if inverse_temp_start > inverse_temp_max:
        raise ValueError("--inverse-temp-start must be <= --inverse-temp-max.")
    if multi_label_mode not in {"any", "all"}:
        raise ValueError(
            f"Unknown Rashomon multi-label mode: {multi_label_mode!r}. Expected 'any' or 'all'."
        )
    from src.IntervalTensor import IntervalTensor
    from src.verification import verify

    resolved_surrogate = verify.resolve_surrogate_form(multi_label_mode, surrogate)
    device_t = torch.device(device)
    states = dataset["state"].to(device_t)
    masks = dataset["actions"].to(device_t)
    if not bool(masks.bool().any(dim=1).all().item()):
        raise ValueError("Dataset contains a state with no valid actions.")
    model.eval()
    with torch.no_grad():
        logits = model(states)
        point_logits = IntervalTensor(logits, logits)
        calibration_value = float("-inf")
        calibration_threshold = 0.0
        min_margin = float("-inf")
        valid_counts = masks.bool().sum(dim=1).to(dtype=logits.dtype)
        state_thresholds = valid_counts / (1.0 + valid_counts)
        for inverse_temp in range(int(inverse_temp_start), int(inverse_temp_max) + 1):
            margins = verify.bound_multi_label_accuracy_margin(
                point_logits,
                masks,
                tau=1.0 / float(inverse_temp),
                lower=True,
                aggregation="none",
                mode=multi_label_mode,
                surrogate=surrogate,
            )
            min_margin = float(margins.min().item())
            if resolved_surrogate == "probability" and multi_label_mode == "any":
                valid_mass = (
                    torch.softmax(logits * inverse_temp, dim=1) * masks
                ).sum(dim=1)
                min_valid_mass, min_mass_index = valid_mass.min(dim=0)
                calibration_value = float(min_valid_mass.item())
                calibration_threshold = float(state_thresholds[min_mass_index].item())
            else:
                calibration_value = min_margin
                calibration_threshold = 0.0
            feasible = (
                min_margin >= 0.0
                if resolved_surrogate == "probability" and multi_label_mode == "any"
                else min_margin > 0.0
            )
            if feasible:
                return int(inverse_temp), calibration_value, calibration_threshold
    raise ValueError(
        "Could not calibrate inverse temperature for Rashomon surrogate "
        f"({resolved_surrogate}): min_state_specific_margin={min_margin:.6f}.",
    )


def compute_rashomon_bounds(
    model: nn.Sequential,
    dataset: dict[str, torch.Tensor],
    *,
    seed: int,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    inverse_temp: int,
    growth_method: str = "IBP",
    growth_method_kwargs: dict[str, Any] | None = None,
    certification_method: str = "IBP",
    multi_label_mode: str = "any",
    surrogate: str = "auto",
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    """Run IntervalTrainer and select a 100%-certified Rashomon box.

    ``growth_method`` selects the verification backend that actually drives box
    growth (both the differentiable soft surrogate and the hard per-iteration
    accuracy check the Lagrangian penalises against). Defaults to ``"IBP"``,
    matching the historical behaviour: cheap, but conservative fast with
    depth, so the box can never grow past what IBP's own bound is willing to
    certify along the way. A tighter method (e.g. ``"CROWN"``) lets the
    optimizer keep growing into regions IBP would have prematurely penalised,
    at the cost of a more expensive bound per iteration.

    ``certification_method`` selects the verification backend used to compute
    the *reported* checkpoint certificates (see ``src.verification.registry``
    for the registered options, e.g. ``"IBP"``, ``"CROWN"``, ``"alpha-CROWN"``).
    This is independent of ``growth_method`` and only changes which checkpoints
    get confirmed as fully certified (``min_hard_acc >= 1.0``) when their
    ``(param_l, param_u)`` are re-verified after the fact -- it cannot recover
    a bigger box than whatever ``growth_method``'s trajectory actually explored.
    """

    from src.trainer.IntervalTrainer import IntervalTrainer

    tensor_dataset = TensorDataset(dataset["state"], dataset["actions"])
    interval_trainer = IntervalTrainer(
        model=model,
        accuracy=1.0,
        min_acc_increment=0,
        seed=int(seed),
        n_certificate_samples=int(certificate_samples),
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        batch_size=int(batch_size),
    )
    interval_trainer.compute_rashomon_set(
        dataset=tensor_dataset,
        temperatures={None: 1.0 / float(inverse_temp)},
        growth_method=growth_method,
        growth_method_kwargs=growth_method_kwargs,
        certification_method=certification_method,
        multi_label_mode=multi_label_mode,
        surrogate=surrogate,
    )
    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in interval_trainer.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        raise ValueError(f"No Rashomon certificate reached 1.0; certificates={cert_values}.")

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [param.detach().cpu() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().cpu() for param in bounded_model.param_u]
    metadata = {
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(value) for value in cert_values],
        "iterations_run": int(n_iters),
        "temperatures": {str(key): float(value) for key, value in interval_trainer.temperatures.items()},
        "multi_label_mode": multi_label_mode,
        "surrogate": interval_trainer.surrogate,
        "resolved_surrogate": interval_trainer.resolved_surrogate,
    }
    return param_bounds_l, param_bounds_u, bounded_model, metadata


def compute_zonotope_region(
    model: nn.Sequential,
    dataset: dict[str, torch.Tensor],
    *,
    seed: int,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    inverse_temp: int,
    rank: int | None,
    multi_label_mode: str = "any",
    surrogate: str = "auto",
) -> tuple[object, object, dict[str, Any]]:
    """Run the zonotope safe-region engine and select a certified region."""

    from src.zonotope_rashomon import (
        compute_zonotope_rashomon_set,
        select_certified_zonotope,
    )

    tensor_dataset = TensorDataset(dataset["state"], dataset["actions"])
    result = compute_zonotope_rashomon_set(
        model,
        tensor_dataset,
        rank=rank,
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        batch_size=int(batch_size),
        certificate_samples=int(certificate_samples),
        inverse_temp=int(inverse_temp),
        seed=int(seed),
        multi_label_mode=multi_label_mode,  # type: ignore[arg-type]
        surrogate=surrogate,  # type: ignore[arg-type]
    )
    selected = select_certified_zonotope(result)
    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in result.certificates
    ]
    if selected is None:
        raise ValueError(f"No zonotope Rashomon certificate reached 1.0; certificates={cert_values}.")
    region, selected_idx = selected
    metadata = {
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(value) for value in cert_values],
        "temperatures": {str(key): float(value) for key, value in result.temperatures.items()},
        "multi_label_mode": multi_label_mode,
        "surrogate": result.surrogate,
        "resolved_surrogate": result.resolved_surrogate,
        "zonotope_rank": int(region.generators.shape[0]),
    }
    return region, result, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a base policy from a saved shield and compute a Rashomon set.",
    )
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, default=None)
    # When given, the BC base policy (and hence the PSPO actor) is fitted on the
    # env's decoded feature representation instead of a one-hot state id, so it
    # matches the observation the deployed PSPO actor receives.
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", default=None)
    parser.add_argument(
        "--state-representation",
        choices=("features", "one_hot", "state_id_lookup"),
        default="features",
        help=(
            "BC/certificate input representation. 'features' requires --env-id; "
            "'state_id_lookup' is exactly equivalent to one-hot but stores integer IDs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--base-policy-path",
        type=Path,
        default=None,
        help=(
            "Load this exact saved base_policy.pt and grow the safe region around "
            "it instead of fitting a new BC policy. Its architecture, state "
            "representation, safety, and requested BC objective are validated."
        ),
    )
    parser.add_argument(
        "--base-policy-only",
        action="store_true",
        help=(
            "Fit or validate the base policy and save its dataset artifacts, but "
            "do not calibrate a Rashomon temperature or grow a safe region. This "
            "is intended for directional PSPO, whose first proposal "
            "determines the initial region-growth direction."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=2,
        help=(
            "Hidden layers in the BC base policy, and hence in the PSPO actor and "
            "critic (SB3 applies a flat net_arch to both). Default 2 matches the "
            "[64, 64] MLP every baseline uses, so PSPO is compared like-for-like. "
            "0 gives the older linear/tabular actor."
        ),
    )
    # Feature-based BC needs more optimisation than one-hot to reach the safety
    # margin: nearby feature points can require different safe actions, so the
    # decision boundary is finer. These defaults suit both representations
    # (one-hot converges well within them and stops early).
    parser.add_argument("--bc-lr", type=float, default=3e-3)
    parser.add_argument("--bc-max-epochs", type=int, default=8000)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--linear-init-margin", type=float, default=BC_TARGET_MARGIN)
    parser.add_argument(
        "--bc-initialisation-objective",
        choices=("margin", "safe_mass"),
        default=BC_INITIALISATION_OBJECTIVE,
        help=(
            "Base-policy fitting objective. 'margin' preserves the legacy logit-"
            "margin loss. 'safe_mass' uses aggregate safe/unsafe probability-mass "
            "cross entropy plus a uniformity loss among safe actions."
        ),
    )
    parser.add_argument(
        "--bc-target-margin",
        type=float,
        default=BC_TARGET_MARGIN,
        help=(
            "Minimum required gap between the best safe and best unsafe logit in "
            "every state when --bc-initialisation-objective=margin. Matches "
            "--linear-init-margin so the gradient path (used "
            "whenever the base policy has hidden layers) reaches the same "
            "separation the closed-form linear initialiser writes directly. "
            "Stopping at bare 100%% accuracy leaves a knife-edge policy whose "
            "greedy actions flip on the first gradient step downstream."
        ),
    )
    parser.add_argument(
        "--bc-margin-loss-weight",
        type=parse_nonnegative_finite_float,
        default=1.0,
        help="Weight of the margin hinge added to the BC loss. 0 disables it.",
    )
    parser.add_argument(
        "--bc-margin-mode",
        choices=("any", "all"),
        default=BC_MARGIN_MODE,
        help=(
            "BC safety-margin semantics. 'any' requires the best safe action "
            "logit to beat the best unsafe action logit. 'all' requires every "
            "safe action logit to beat the best unsafe action logit."
        ),
    )
    parser.add_argument(
        "--bc-safe-action-entropy-weight",
        type=parse_nonnegative_finite_float,
        default=BC_SAFE_ACTION_ENTROPY_WEIGHT,
        help=(
            "Weight of the reward-free entropy regularizer on the policy "
            "distribution conditioned on safe actions. Enabled by default."
        ),
    )
    parser.add_argument(
        "--bc-min-safe-action-entropy",
        type=parse_unit_interval_float,
        default=BC_MIN_SAFE_ACTION_ENTROPY,
        help=(
            "Minimum normalized conditional safe-action entropy required in every "
            "state with multiple safe actions when the entropy weight is positive."
        ),
    )
    parser.add_argument(
        "--bc-unsafe-mass-target",
        type=parse_unsafe_mass_target,
        default=BC_UNSAFE_MASS_TARGET,
        help=(
            "Finite target probability mass assigned to unsafe actions by the "
            "safe_mass objective. Must be strictly between 0 and 0.5."
        ),
    )
    parser.add_argument(
        "--bc-max-unsafe-mass",
        type=parse_max_unsafe_mass,
        default=BC_MAX_UNSAFE_MASS,
        help=(
            "Maximum unsafe probability mass allowed in every training state "
            "before safe_mass initialisation stops."
        ),
    )
    parser.add_argument(
        "--bc-safe-action-uniformity-weight",
        type=parse_nonnegative_finite_float,
        default=BC_SAFE_ACTION_UNIFORMITY_WEIGHT,
        help=(
            "Weight of KL(uniform || policy conditioned on safe actions) in the "
            "safe_mass objective."
        ),
    )
    parser.add_argument("--no-direct-linear-init", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=2000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument(
        "--rashomon-batch-size",
        type=parse_rashomon_batch_size,
        default="auto",
        help=(
            "Safe-behaviour optimisation batch size. 'auto' (default) uses the "
            "entire safe-behaviour demonstration dataset; a positive integer "
            "requests an explicit size."
        ),
    )
    parser.add_argument("--certificate-samples", type=int, default=1000)
    parser.add_argument(
        "--safe-region-shape",
        choices=("orthotope", "zonotope"),
        default="orthotope",
        help="Safe parameter-region geometry to compute for PSPO.",
    )
    parser.add_argument(
        "--zonotope-rank",
        type=int,
        default=None,
        help="Number of learned zonotope generator directions. Defaults to min(16, n_actor_params).",
    )
    parser.add_argument("--inverse-temp-start", type=int, default=1)
    parser.add_argument("--inverse-temp-max", type=int, default=1000)
    parser.add_argument(
        "--rashomon-multi-label-mode",
        choices=("any", "all"),
        default=RASHOMON_MULTI_LABEL_MODE,
        help=(
            "Admissible-set certificate/surrogate used for the Rashomon safe "
            "parameter set. 'any' requires at least one safe action logit to "
            "beat every unsafe action logit. 'all' requires every safe action "
            "logit to beat every unsafe action logit."
        ),
    )
    parser.add_argument(
        "--rashomon-surrogate",
        choices=("auto", "probability", "logsumexp"),
        default=RASHOMON_SURROGATE,
        help=(
            "Soft constraint used while growing the Rashomon region. 'auto' "
            "preserves the historical formula for each multi-label mode; "
            "'logsumexp' uses the temperature-scaled LSE margin for both modes."
        ),
    )
    parser.add_argument(
        "--growth-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verification backend that actually drives box growth (the "
            "differentiable soft surrogate and the hard per-iteration "
            "accuracy check the Lagrangian penalises against). Defaults to "
            "IBP: cheap, but conservative fast with depth, so the box can "
            "never grow past what IBP's own bound is willing to certify "
            "along the way. A tighter method (e.g. CROWN) lets the "
            "optimizer keep growing into regions IBP would have "
            "prematurely penalised, at the cost of a slower bound per "
            "iteration. Independent of --certification-method."
        ),
    )
    parser.add_argument(
        "--certification-method",
        choices=("IBP", "CROWN", "alpha-CROWN"),
        default="IBP",
        help=(
            "Verification backend used for the reported Rashomon checkpoint "
            "certificates (see src.verification.registry), independent of "
            "--growth-method. Only affects which checkpoints get confirmed "
            "as fully certified; it cannot recover a bigger box than "
            "--growth-method's trajectory actually explored."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy_initialisation_started = time.perf_counter()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mask = load_shield_mask(args.shield_path, risk_threshold=args.risk_threshold)
    state_to_features = None
    if args.state_representation == "features":
        if not args.env_id:
            raise ValueError("--state-representation features requires --env-id.")
        from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs
        from projects.safe_policy_optimisation.utils.safe_crl_bridge import (
            make_custom_masa_env,
        )

        feature_env = make_custom_masa_env(
            args.env_id,
            max_episode_steps=None,
            env_kwargs=parse_env_kwargs(args.env_kwargs),
        ).unwrapped
        state_to_features = feature_env.state_to_features
    elif args.state_representation == "state_id_lookup":
        unsupported = []
        if args.growth_method != "IBP":
            unsupported.append(f"growth_method={args.growth_method}")
        if args.certification_method != "IBP":
            unsupported.append(f"certification_method={args.certification_method}")
        if args.safe_region_shape != "orthotope":
            unsupported.append(f"safe_region_shape={args.safe_region_shape}")
        if unsupported:
            raise ValueError(
                "state_id_lookup currently supports IBP orthotopes only; got "
                + ", ".join(unsupported)
                + "."
            )
    dataset, dataset_metadata = make_safe_behaviour_payload(
        mask,
        state_to_features,
        state_representation=args.state_representation,
    )
    rashomon_batch_size = resolve_rashomon_batch_size(args.rashomon_batch_size, mask)
    input_dim = int(dataset_metadata["input_dim"])
    n_actions = int(dataset["actions"].shape[1])

    base_policy_source = None
    if args.base_policy_path is None:
        model = build_base_policy(
            input_dim,
            n_actions,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            state_id_lookup=args.state_representation == "state_id_lookup",
        )
        bc_metrics = fit_base_policy(
            model,
            dataset,
            lr=args.bc_lr,
            max_epochs=args.bc_max_epochs,
            batch_size=args.bc_batch_size,
            seed=args.seed,
            device=args.device,
            direct_linear_init=not args.no_direct_linear_init,
            linear_init_margin=args.linear_init_margin,
            target_margin=args.bc_target_margin,
            margin_loss_weight=args.bc_margin_loss_weight,
            margin_mode=args.bc_margin_mode,
            safe_action_entropy_weight=args.bc_safe_action_entropy_weight,
            min_safe_action_entropy=args.bc_min_safe_action_entropy,
            initialisation_objective=args.bc_initialisation_objective,
            unsafe_mass_target=args.bc_unsafe_mass_target,
            max_unsafe_mass=args.bc_max_unsafe_mass,
            safe_action_uniformity_weight=args.bc_safe_action_uniformity_weight,
        )
    else:
        model, bc_metrics, base_policy_source = load_base_policy_for_dataset(
            args.base_policy_path,
            dataset,
            dataset_metadata,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            target_margin=args.bc_target_margin,
            margin_loss_weight=args.bc_margin_loss_weight,
            margin_mode=args.bc_margin_mode,
            safe_action_entropy_weight=args.bc_safe_action_entropy_weight,
            min_safe_action_entropy=args.bc_min_safe_action_entropy,
            initialisation_objective=args.bc_initialisation_objective,
            unsafe_mass_target=args.bc_unsafe_mass_target,
            max_unsafe_mass=args.bc_max_unsafe_mass,
            safe_action_uniformity_weight=args.bc_safe_action_uniformity_weight,
            device=args.device,
        )
    if not bc_metrics["reached_target"]:
        entropy_requirement = ""
        entropy_enabled = (
            float(bc_metrics["safe_action_uniformity_weight"]) > 0.0
            if bc_metrics["initialisation_objective"] == "safe_mass"
            else float(bc_metrics["safe_action_entropy_weight"]) > 0.0
        )
        if entropy_enabled:
            entropy_requirement = (
                ", final_min_safe_action_entropy="
                f"{bc_metrics['final_normalized_safe_action_entropy_min']:.4f} "
                f"(target {bc_metrics['min_safe_action_entropy']:.4f})"
            )
        if bc_metrics["initialisation_objective"] == "safe_mass":
            target_description = (
                "maximum unsafe action mass: "
                f"final_accuracy={bc_metrics['final_accuracy']:.6f}, "
                "final_max_unsafe_action_mass="
                f"{bc_metrics['final_max_unsafe_action_mass']:.6f} "
                f"(maximum {bc_metrics['max_unsafe_mass']:.6f})"
            )
        else:
            if float(bc_metrics["margin_loss_weight"]) == 0.0:
                mode = str(bc_metrics["bc_margin_mode"])
                label = "strict all-safe" if mode == "all" else "any-safe"
                margin_key = "final_min_all_margin" if mode == "all" else "final_min_any_margin"
                target_description = (
                    f"{label} feasibility: "
                    f"final_accuracy={bc_metrics['final_accuracy']:.6f}, "
                    f"{margin_key}="
                    f"{bc_metrics[margin_key]:.4f} (required > 0)"
                )
            else:
                target_description = (
                    f"{bc_metrics['bc_margin_mode']!r} safety margin: "
                    f"final_accuracy={bc_metrics['final_accuracy']:.6f}, "
                    f"final_min_margin={bc_metrics['final_min_margin']:.4f} "
                    f"(target {bc_metrics['target_margin']:.4f})"
                )
        raise RuntimeError(
            "Base policy did not reach 100% allowed-action accuracy at the required "
            f"{target_description}{entropy_requirement}. Raise --bc-max-epochs or "
            "--bc-lr, or relax the requested targets.",
        )

    safe_dataset_path = run_dir / "safe_behaviour_dataset.pt"
    rashomon_dataset_path = run_dir / "rashomon_dataset.pt"
    base_policy_path = run_dir / "base_policy.pt"
    bounded_model_path = run_dir / "rashomon_bounded_model.pt"
    bounds_path = run_dir / "rashomon_param_bounds.pt"
    zonotope_path = run_dir / "rashomon_zonotope_region.pt"
    architecture = {
        "input_dim": input_dim,
        "n_actions": n_actions,
        "hidden_dim": int(args.hidden_dim),
        "n_hidden": int(args.n_hidden),
        "activation": "Tanh",
        "state_representation": dataset_metadata["state_representation"],
    }

    torch.save(dataset, safe_dataset_path)
    torch.save(dataset, rashomon_dataset_path)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": architecture,
            "bc_metrics": bc_metrics,
        },
        base_policy_path,
    )

    common_summary = {
        "shield_path": str(args.shield_path),
        "shield_sha256": _file_sha256(args.shield_path),
        "run_dir": str(run_dir),
        "safe_behaviour_dataset_path": str(safe_dataset_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "base_policy_path": str(base_policy_path),
        "base_policy_source": base_policy_source,
        "architecture": architecture,
        "dataset": dataset_metadata,
        "base_policy": bc_metrics,
        "timing": {
            "policy_initialisation_wall_time_s": float(
                time.perf_counter() - policy_initialisation_started
            ),
        },
    }
    if args.base_policy_only:
        summary = {
            **common_summary,
            "base_policy_only": True,
            "rashomon": {
                "status": "skipped",
                "reason": "base_policy_only",
            },
        }
        write_json(run_dir / "summary.json", summary)
        log_info(f"Base-policy artifacts written to {run_dir}")
        return summary

    inverse_temp, min_valid_mass, surrogate_threshold = calibrate_inverse_temperature(
        model,
        dataset,
        inverse_temp_start=args.inverse_temp_start,
        inverse_temp_max=args.inverse_temp_max,
        device=args.device,
        multi_label_mode=args.rashomon_multi_label_mode,
        surrogate=args.rashomon_surrogate,
    )
    n_params = sum(param.numel() for param in model.parameters())
    zonotope_rank = (
        zonotope_rank_default(n_params)
        if args.zonotope_rank is None
        else int(args.zonotope_rank)
    )
    if args.safe_region_shape == "orthotope":
        param_bounds_l, param_bounds_u, bounded_model, rashomon_metadata = compute_rashomon_bounds(
            model,
            dataset,
            seed=args.seed,
            n_iters=args.rashomon_n_iters,
            checkpoint=args.rashomon_checkpoint,
            batch_size=rashomon_batch_size,
            certificate_samples=args.certificate_samples,
            inverse_temp=inverse_temp,
            growth_method=args.growth_method,
            certification_method=args.certification_method,
            multi_label_mode=args.rashomon_multi_label_mode,
            surrogate=args.rashomon_surrogate,
        )
        zonotope_region = None
        zonotope_result = None
    else:
        zonotope_region, zonotope_result, rashomon_metadata = compute_zonotope_region(
            model,
            dataset,
            seed=args.seed,
            n_iters=args.rashomon_n_iters,
            checkpoint=args.rashomon_checkpoint,
            batch_size=rashomon_batch_size,
            certificate_samples=args.certificate_samples,
            inverse_temp=inverse_temp,
            rank=zonotope_rank,
            multi_label_mode=args.rashomon_multi_label_mode,
            surrogate=args.rashomon_surrogate,
        )
        param_bounds_l = None
        param_bounds_u = None
        bounded_model = None

    if args.safe_region_shape == "orthotope":
        torch.save(bounded_model, bounded_model_path)
        torch.save({"param_bounds_l": param_bounds_l, "param_bounds_u": param_bounds_u}, bounds_path)
        safe_action_logit_analysis = safe_action_logit_interval_analysis(
            bounded_model,
            dataset,
            device=args.device,
        )
    else:
        torch.save(zonotope_result, bounded_model_path)
        torch.save(
            {
                "safe_region_shape": "zonotope",
                "center_params": zonotope_region.center_params,
                "generators": zonotope_region.generators,
                "coefficient_l": zonotope_region.coefficient_l,
                "coefficient_u": zonotope_region.coefficient_u,
                "param_shapes": zonotope_region.param_shapes,
                "zonotope_rank": int(zonotope_region.generators.shape[0]),
            },
            zonotope_path,
        )
        safe_action_logit_analysis = {
            "status": "not_computed",
            "reason": "safe-action logit interval analysis is currently implemented for orthotope bounded models.",
        }

    summary = {
        **common_summary,
        "base_policy_only": False,
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "rashomon_zonotope_region_path": str(zonotope_path),
        "rashomon": {
            "safe_region_shape": args.safe_region_shape,
            "zonotope_rank": int(zonotope_rank) if args.safe_region_shape == "zonotope" else None,
            "inverse_temperature": int(inverse_temp),
            "min_valid_mass": (
                float(min_valid_mass)
                if args.rashomon_multi_label_mode == "any"
                and rashomon_metadata["resolved_surrogate"] == "probability"
                else None
            ),
            "min_all_safe_margin": float(min_valid_mass) if args.rashomon_multi_label_mode == "all" else None,
            "min_lse_margin": (
                float(min_valid_mass) if args.rashomon_surrogate == "logsumexp" else None
            ),
            "surrogate_threshold": float(surrogate_threshold),
            "multi_label_mode": args.rashomon_multi_label_mode,
            "surrogate": args.rashomon_surrogate,
            "resolved_surrogate": rashomon_metadata["resolved_surrogate"],
            "n_iters": int(args.rashomon_n_iters),
            "checkpoint": int(args.rashomon_checkpoint),
            "batch_size_setting": args.rashomon_batch_size,
            "batch_size": int(rashomon_batch_size),
            "certificate_samples": int(args.certificate_samples),
            "growth_method": args.growth_method,
            "certification_method": args.certification_method,
            "safe_action_logit_analysis": safe_action_logit_analysis,
            **rashomon_metadata,
        },
    }
    write_json(run_dir / "summary.json", summary)
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
