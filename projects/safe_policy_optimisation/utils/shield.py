"""Loading binary safety shields from stored ``shield_q.pt`` artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_torch_payload(path: Path) -> Any:
    """Load a torch artifact onto CPU (full unpickling, weights_only=False)."""

    return torch.load(path, map_location="cpu", weights_only=False)


def load_shield_mask(
    shield_path: Path,
    *,
    shield_key: str = "shield",
    source: str = "shield",
    risk_threshold: float | None = None,
    dtype: np.dtype | type = np.int64,
) -> np.ndarray:
    """Load a binary ``(state, action)`` shield mask from a shield artifact.

    ``source`` selects how the mask is derived:

    - ``"shield"``: read the precomputed boolean mask under ``shield_key``.
    - ``"action_risk"``: threshold the stored ``action_risk`` table at
      ``risk_threshold`` (falling back to the artifact's ``risk_threshold``,
      else ``0.0``).
    - ``"auto"``: use ``shield_key`` if present, otherwise ``action_risk``.

    The returned array is cast to ``dtype`` (``int64`` for shielded-PPO masks,
    ``float32`` for Rashomon one-hot feature construction).
    """

    payload = load_torch_payload(shield_path)
    resolved = source
    if resolved == "auto":
        resolved = "shield" if shield_key in payload else "action_risk"

    if resolved == "shield":
        if shield_key not in payload:
            raise KeyError(
                f"Shield artifact has no key {shield_key!r}; keys={sorted(payload.keys())}"
            )
        mask = payload[shield_key]
    elif resolved == "action_risk":
        if "action_risk" not in payload:
            raise KeyError("Shield artifact has no 'action_risk' key.")
        threshold = (
            payload.get("risk_threshold", 0.0)
            if risk_threshold is None
            else risk_threshold
        )
        mask = np.asarray(_as_numpy(payload["action_risk"])) <= float(threshold)
    else:
        raise ValueError(f"Unknown shield source {source!r}.")

    mask = _as_numpy(mask)
    if mask.ndim != 2:
        raise ValueError(f"Shield mask must be 2-D, got shape {mask.shape}.")
    return (mask != 0).astype(dtype)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
