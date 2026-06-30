"""Single adapter for the cross-project dependency on ``projects.safe_crl``.

Safe-policy-optimisation reuses pieces of the sibling ``safe_crl`` project
(MASA-style environment registration and the safety-retention shield synthesis
entry points). Routing those imports through this one module keeps the coupling
explicit and in a single place, instead of repeating the cross-project import in
every stage.
"""

from __future__ import annotations

from projects.safe_crl.utils.masa_tabular_envs.factory import make_custom_masa_env

__all__ = [
    "make_custom_masa_env",
    "register_masa_envs",
]


def register_masa_envs() -> bool:
    """Best-effort registration of the local ``Custom*`` MASA Gymnasium ids.

    Returns ``True`` if the registration module was importable, ``False`` if the
    optional ``safe_crl`` env package is unavailable. Importing the module has
    the side effect of registering the environment ids with Gymnasium.
    """

    try:
        import projects.safe_crl.utils.masa_tabular_envs  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
