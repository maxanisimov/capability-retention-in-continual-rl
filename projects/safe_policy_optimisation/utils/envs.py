"""Environment-kwargs parsing shared across stage scripts."""

from __future__ import annotations

import argparse
import json
from typing import Any

from projects.safe_policy_optimisation.utils.safe_crl_bridge import register_masa_envs

__all__ = ["parse_env_kwargs", "env_kwargs_from_args", "register_masa_envs"]

GHOST_RAND_ENV_ID = "CustomMiniPacman-v0"


def parse_env_kwargs(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalise ``--env-kwargs`` into a dict.

    Accepts ``None`` (→ ``{}``), an already-decoded mapping (as the pipeline
    passes from YAML), or a JSON-object string (as the CLI passes).
    """

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--env-kwargs must decode to a JSON object.")
    return payload


def env_kwargs_from_args(
    args: argparse.Namespace, *, ghost_rand_env_id: str = GHOST_RAND_ENV_ID
) -> dict[str, Any]:
    """Resolve env kwargs from parsed args, defaulting MiniPacman's ghost prob."""

    if getattr(args, "env_kwargs", None) is not None:
        return parse_env_kwargs(args.env_kwargs)
    if args.env_id == ghost_rand_env_id and hasattr(args, "ghost_rand_prob"):
        return {"ghost_rand_prob": float(args.ghost_rand_prob)}
    return {}
