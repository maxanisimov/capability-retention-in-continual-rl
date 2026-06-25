"""Safe initialization for provably-safe policy optimisation.

A randomly initialised policy proposes unsafe actions; the shield masks this during
training, but the policy itself is unsafe (so unshielded ``predict`` is unsafe and the
shield intervenes constantly). These helpers make the policy *itself* propose safe
actions before RL training:

1. **Behavioural cloning (BC)** on shield data -- a supervised loss that puts the policy's
   probability mass on the shield's safe actions until ``argmax`` is safe (the loss form
   used by the FrozenLake source-training safety fine-tune).
2. **Certified refinement** -- train the worst-case (interval-bound-propagation) margin so
   that the greedy action is provably safe over each region box, then confirm with the
   project's trusted verifier (``src.verification.api.verify_dataset``).

The differentiable IBP forward here is standard, sound interval propagation for
Linear/ReLU/Tanh/Flatten networks; it agrees with the official verifier (both are IBP), so
driving its margin positive makes ``verify_dataset`` certify.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

LogitsFn = Callable[[th.Tensor], th.Tensor]


@dataclasses.dataclass
class SafeInitReport:
    """Outcome of :func:`pretrain_on_shield`."""

    bc_epochs: int
    sampled_greedy_safe_rate: float
    refine_epochs: int = 0
    certified_fraction: float | None = None  # None when certification was not run
    all_certified: bool | None = None


def _ibp_bounds(seq: nn.Sequential, x_l: th.Tensor, x_u: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Differentiable interval bound propagation through a Linear/ReLU/Tanh/Flatten net."""
    lower, upper = x_l, x_u
    for module in seq:
        if isinstance(module, nn.Linear):
            center = (lower + upper) / 2
            radius = (upper - lower) / 2
            out_center = F.linear(center, module.weight, module.bias)
            out_radius = F.linear(radius, module.weight.abs(), None)
            lower, upper = out_center - out_radius, out_center + out_radius
        elif isinstance(module, nn.ReLU):
            lower, upper = F.relu(lower), F.relu(upper)
        elif isinstance(module, nn.Tanh):
            lower, upper = th.tanh(lower), th.tanh(upper)
        elif isinstance(module, nn.Flatten):
            lower, upper = th.flatten(lower, 1), th.flatten(upper, 1)
        elif isinstance(module, nn.Identity):
            continue
        else:
            raise TypeError(
                f"IBP certification does not support layer {type(module).__name__}; "
                "supported: Linear, ReLU, Tanh, Flatten.",
            )
    return lower, upper


def _greedy_safe_rate(logits: th.Tensor, safe_mask: th.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = safe_mask[th.arange(safe_mask.shape[0], device=safe_mask.device), preds]
    return float(correct.float().mean().item())


def behavioural_clone(
    logits_fn: LogitsFn,
    observations: th.Tensor,
    safe_mask: th.Tensor,
    params: Sequence[th.nn.Parameter],
    *,
    lr: float,
    max_epochs: int,
    target_rate: float = 1.0,
) -> tuple[int, float]:
    """Maximise probability mass on safe actions until greedy is safe.

    ``safe_mask`` is a boolean ``(n, n_actions)`` mask. Returns ``(epochs_run, rate)``.
    """
    optimizer = th.optim.Adam(params, lr=lr)
    rate = _greedy_safe_rate(logits_fn(observations).detach(), safe_mask)
    if rate >= target_rate:
        return 0, rate
    for epoch in range(1, max_epochs + 1):
        logits = logits_fn(observations)
        masked = logits.masked_fill(~safe_mask, -1e9)
        log_p_safe = th.logsumexp(masked, dim=1) - th.logsumexp(logits, dim=1)
        loss = -log_p_safe.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rate = _greedy_safe_rate(logits_fn(observations).detach(), safe_mask)
        if rate >= target_rate:
            return epoch, rate
    return max_epochs, rate


def _worst_case_margin(
    seq: nn.Sequential,
    x_l: th.Tensor,
    x_u: th.Tensor,
    safe_mask: th.Tensor,
) -> th.Tensor:
    """Per-region worst-case margin ``min_safe lower - max_unsafe upper`` (>0 ⇒ certified)."""
    lower, upper = _ibp_bounds(seq, x_l, x_u)
    neg_inf = th.tensor(float("-inf"), device=lower.device)
    pos_inf = th.tensor(float("inf"), device=lower.device)
    safe_lower = th.where(safe_mask, lower, pos_inf).min(dim=1).values
    unsafe_upper = th.where(~safe_mask, upper, neg_inf).max(dim=1).values
    return safe_lower - unsafe_upper


def certified_refine(
    seq: nn.Sequential,
    x_l: th.Tensor,
    x_u: th.Tensor,
    safe_mask: th.Tensor,
    params: Sequence[th.nn.Parameter],
    *,
    lr: float,
    max_epochs: int,
    target_margin: float = 0.1,
) -> int:
    """Train the worst-case IBP margin over region boxes until certified or budget spent."""
    optimizer = th.optim.Adam(params, lr=lr)
    for epoch in range(1, max_epochs + 1):
        margin = _worst_case_margin(seq, x_l, x_u, safe_mask)
        if float(margin.min().item()) > 0:
            return epoch - 1
        loss = F.relu(target_margin - margin).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return max_epochs


def certify_with_verifier(
    seq: nn.Sequential,
    x_l: th.Tensor,
    x_u: th.Tensor,
    safe_mask: th.Tensor,
    *,
    method: str = "IBP",
) -> tuple[float, bool]:
    """Authoritative greedy-safety certificate via the project's verifier.

    Certifies ``argmax(seq(x')) in safe set`` for all ``x'`` in each ``[x_l, x_u]`` box,
    using ``src.verification.api`` (independent of the differentiable IBP above). Returns
    ``(certified_fraction, all_certified)``.
    """
    from src.verification.api import AdmissibleSet, build_bounded_model, verify_dataset

    bounded_model = build_bounded_model(seq, method)
    result = verify_dataset(
        bounded_model,
        AdmissibleSet(n_classes=int(safe_mask.shape[1]), multi_hot=safe_mask.bool()),
        X_l=x_l,
        X_u=x_u,
    )
    return result.certified_fraction, result.all_certified


def _bc_data(model: Any, shield: Any, n_samples: int, seed: int | None):
    """Return (raw_obs, safe_mask) for behavioural cloning."""
    from gymnasium import spaces

    if isinstance(model.observation_space, spaces.Discrete):
        states = np.arange(int(model.observation_space.n))
        return states, shield.mask[states]
    # Continuous Box: sample observations (finite bounds required).
    low = np.asarray(model.observation_space.low, dtype=np.float64)
    high = np.asarray(model.observation_space.high, dtype=np.float64)
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("Safe-init sampling needs a finite observation space; got non-finite bounds.")
    rng = np.random.default_rng(seed)
    obs = rng.uniform(low, high, size=(n_samples, low.shape[0])).astype(np.float32)
    return obs, shield.mask[shield.obs_to_state(obs)]


def _certification_inputs(model: Any, shield: Any):
    """Return (X_l, X_u, safe_mask) in the network's input space, or None if unavailable.

    Discrete: one-hot point per state (exact). Continuous: each RegionShield box clamped
    to the observation space. Predicate-only RegionShields return None (no IBP boxes).
    """
    from gymnasium import spaces
    from stable_baselines3.common.preprocessing import preprocess_obs

    device = model.device
    if isinstance(model.observation_space, spaces.Discrete):
        states = np.arange(int(model.observation_space.n))
        obs_t, _ = model.policy.obs_to_tensor(states)
        x = preprocess_obs(obs_t, model.observation_space).float()
        mask = th.as_tensor(shield.mask[states], dtype=th.bool, device=device)
        return x, x, mask

    boxes = getattr(shield, "boxes", None)
    if not boxes:
        return None
    low = np.asarray(model.observation_space.low, dtype=np.float32)
    high = np.asarray(model.observation_space.high, dtype=np.float32)
    x_l = np.stack([np.clip(np.where(np.isfinite(lo), lo, low), low, high) for lo, _ in boxes])
    x_u = np.stack([np.clip(np.where(np.isfinite(hi), hi, high), low, high) for _, hi in boxes])
    x_l_t = th.as_tensor(x_l, dtype=th.float32, device=device)
    x_u_t = th.as_tensor(x_u, dtype=th.float32, device=device)
    mask = th.as_tensor(shield.mask[: len(boxes)], dtype=th.bool, device=device)
    return x_l_t, x_u_t, mask


def run_safe_init(
    model: Any,
    *,
    logits_fn: LogitsFn,
    certify_seq: nn.Sequential,
    params: Sequence[th.nn.Parameter],
    n_samples: int = 4096,
    lr: float = 1e-3,
    bc_max_epochs: int = 500,
    refine_max_epochs: int = 500,
    require_certified: bool = False,
    target_margin: float = 0.1,
    seed: int | None = None,
) -> SafeInitReport:
    """Behavioural-clone then certify a policy's greedy action against its shield."""
    shield = model._shield
    if shield is None:
        raise RuntimeError("pretrain_on_shield requires a shield; none is attached.")
    device = model.device

    # --- Behavioural cloning ---
    raw_obs, safe_mask = _bc_data(model, shield, n_samples, seed)
    obs_t, _ = model.policy.obs_to_tensor(raw_obs)
    safe_mask_t = th.as_tensor(safe_mask, dtype=th.bool, device=device)
    bc_epochs, rate = behavioural_clone(
        logits_fn, obs_t, safe_mask_t, list(params), lr=lr, max_epochs=bc_max_epochs
    )

    report = SafeInitReport(bc_epochs=bc_epochs, sampled_greedy_safe_rate=rate)

    # --- Certified refinement + verification ---
    cert = _certification_inputs(model, shield)
    if cert is None:
        warnings.warn(
            "Shield has no box regions; skipping IBP certification (behavioural cloning only). "
            "Use RegionShield.from_boxes(...) or a discrete Shield to enable certification.",
            stacklevel=2,
        )
        return report

    x_l, x_u, cert_mask = cert
    report.refine_epochs = certified_refine(
        certify_seq, x_l, x_u, cert_mask, list(params),
        lr=lr, max_epochs=refine_max_epochs, target_margin=target_margin,
    )
    frac, all_certified = certify_with_verifier(certify_seq, x_l, x_u, cert_mask)
    report.certified_fraction = frac
    report.all_certified = all_certified
    if require_certified and not all_certified:
        raise RuntimeError(
            f"Safe init failed to certify all regions (certified_fraction={frac:.3f}). "
            "Try smaller/tighter region boxes, more refine epochs, or a larger network.",
        )
    return report
