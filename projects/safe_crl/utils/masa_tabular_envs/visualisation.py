"""Shield visualisation helpers for custom MASA-style tabular environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.safe_crl.utils.masa_tabular_envs.dynamics import ACTION_MAP, PACMAN_ACTION_MAP
from projects.safe_crl.utils.masa_tabular_envs.frozen_lake import CustomFrozenLake
from projects.safe_crl.utils.masa_tabular_envs.gridworlds import CustomColourBombGridWorldV3
from projects.safe_crl.utils.masa_tabular_envs.media_streaming import (
    CustomMediaStreaming,
    CustomMediaStreamingV2,
    CustomMediaStreamingV3,
)
from projects.safe_crl.utils.masa_tabular_envs.pacman import CustomMiniPacman, CustomPacman

FROZEN_ACTION_NAMES = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}
FROZEN_ACTION_DELTAS = {
    0: (-0.30, 0.00),
    1: (0.00, 0.30),
    2: (0.30, 0.00),
    3: (0.00, -0.30),
}
GRID_ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "down",
    3: "up",
    4: "stay",
}
PACMAN_ACTION_NAMES = GRID_ACTION_NAMES
MEDIA_ACTION_NAMES = {
    0: "slow",
    1: "fast",
}
MEDIA_ACTION_DELTAS = {
    0: (-0.30, 0.00),
    1: (0.30, 0.00),
}


@dataclass(frozen=True)
class _Cell:
    state: int
    row: int
    col: int
    description: str
    center_x: float | None = None
    center_y: float | None = None
    span_w: float | None = None
    span_h: float | None = None


@dataclass(frozen=True)
class _ShieldLayout:
    background: np.ndarray | None
    nrow: int
    ncol: int
    cells: tuple[_Cell, ...]
    action_deltas: Mapping[int, tuple[float, float]]
    action_names: Mapping[int, str]
    title: str


def plot_tabular_shield(
    env: Any,
    shield: np.ndarray,
    *,
    fixed_features: Mapping[str, Any] | None = None,
    info: Any | None = None,
    show_risk: bool = False,
    title: str | None = None,
    ax: Any | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Plot allowed shield actions over a custom MASA tabular environment.

    Args:
        env: Custom MASA-style tabular environment or Gymnasium wrapper.
        shield: Binary mask with shape ``(n_states, n_actions)``.
        fixed_features: Non-position state features to hold fixed when plotting
            a spatial or one-dimensional slice.
        info: Optional shield synthesis info. If it exposes ``action_risk``,
            those risks can be shown by setting ``show_risk=True``.
        show_risk: Colour arrows by action risk and print risk labels.
        title: Optional plot title. Defaults to a concise environment/slice name.
        ax: Optional Matplotlib axes to draw into.
        save_path: Optional path where the figure should be saved.

    Returns:
        The Matplotlib axes used for the plot.
    """

    plt = _require_pyplot()
    unwrapped = _unwrap(env)
    shield = _validate_shield(unwrapped, shield)
    layout = _build_layout(unwrapped, fixed_features, render_env=env)
    action_risk = _action_risk(info, shield.shape) if show_risk else None

    if ax is None:
        _, ax = plt.subplots(figsize=_figure_size(layout))

    if layout.background is None:  # pragma: no cover - internal defensive check
        raise RuntimeError("Expected a rendered tabular shield background.")

    ax.imshow(layout.background)
    ax.set_title(layout.title if title is None else title)
    ax.axis("off")

    height, width = layout.background.shape[:2]
    cell_w = width / float(layout.ncol)
    cell_h = height / float(layout.nrow)

    for cell in layout.cells:
        allowed_actions = np.flatnonzero(shield[cell.state])
        for action in allowed_actions:
            risk = None
            if action_risk is not None:
                risk = float(action_risk[cell.state, int(action)])
            color = _action_color(plt, risk)
            _draw_action(
                ax,
                cell,
                int(action),
                layout,
                cell_w,
                cell_h,
                color,
                risk,
                show_risk,
            )

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax


def build_shield_layout(
    env: Any,
    *,
    fixed_features: Mapping[str, Any] | None = None,
) -> "_ShieldLayout":
    """Return the rendered background plus per-cell / per-action layout for a shield slice.

    The returned object exposes ``background`` (the env's rgb_array frame), ``cells``
    (each with ``state`` and screen geometry), ``action_deltas`` (per-action screen
    direction as a fraction of a cell), ``action_names``, ``nrow``, ``ncol`` and ``title``.
    This is the public entry point external plotters use to overlay their own per-action
    annotations (e.g. an eventual-safety-probability colormap) on the native frame.
    """

    return _build_layout(
        _unwrap(env),
        _normalise_fixed_features(fixed_features),
        render_env=env,
    )


def render_tabular_shield_background(
    env: Any,
    *,
    fixed_features: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Render the background image used by ``plot_tabular_shield``."""

    background = _build_layout(
        _unwrap(env),
        fixed_features,
        render_env=env,
    ).background
    if background is None:  # pragma: no cover - internal defensive check
        raise RuntimeError("Expected a rendered tabular shield background.")
    return background


def print_allowed_actions(
    env: Any,
    shield: np.ndarray,
    *,
    fixed_features: Mapping[str, Any] | None = None,
) -> None:
    """Print allowed actions for the visible states in a shield slice."""

    unwrapped = _unwrap(env)
    shield = _validate_shield(unwrapped, shield)
    layout = _build_layout(unwrapped, fixed_features, render_background=False)

    for cell in layout.cells:
        allowed = [
            layout.action_names.get(int(action), f"action {int(action)}")
            for action in np.flatnonzero(shield[cell.state])
        ]
        if not allowed:
            continue
        labels = sorted(_labels_for_state(unwrapped, cell.state))
        print(
            f"state {cell.state:4d} {cell.description} "
            f"labels={labels}: {allowed}",
        )


def _unwrap(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _require_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
        raise ModuleNotFoundError(
            "matplotlib is required to plot tabular shields.",
        ) from exc
    return plt


def _validate_shield(env: Any, shield: np.ndarray) -> np.ndarray:
    shield = np.asarray(shield)
    expected_shape = (_n_states(env), _n_actions(env))
    if shield.shape != expected_shape:
        raise ValueError(
            f"shield must have shape {expected_shape} for "
            f"{type(env).__name__}, got {shield.shape}.",
        )
    return shield


def _n_states(env: Any) -> int:
    if hasattr(env, "_n_states"):
        return int(env._n_states)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "n"):
        return int(env.observation_space.n)
    raise ValueError(f"Cannot infer tabular state count for {type(env).__name__}.")


def _n_actions(env: Any) -> int:
    if hasattr(env, "_n_actions"):
        return int(env._n_actions)
    if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        return int(env.action_space.n)
    raise ValueError(f"Cannot infer action count for {type(env).__name__}.")


def _build_layout(
    env: Any,
    fixed_features: Mapping[str, Any] | None,
    *,
    render_background: bool = True,
    render_env: Any | None = None,
) -> _ShieldLayout:
    features = _normalise_fixed_features(fixed_features)
    frame_env = env if render_env is None else render_env
    if isinstance(env, CustomFrozenLake):
        return _frozen_lake_layout(env, features, render_background, frame_env)
    if isinstance(env, (CustomMiniPacman, CustomPacman)):
        return _pacman_layout(env, features, render_background, frame_env)
    if isinstance(env, CustomMediaStreamingV3):
        return _media_v3_layout(env, features, render_background, frame_env)
    if isinstance(env, CustomMediaStreamingV2):
        return _media_v2_layout(env, features, render_background, frame_env)
    if isinstance(env, CustomMediaStreaming):
        return _media_v0_layout(env, features, render_background, frame_env)
    if isinstance(env, CustomColourBombGridWorldV3):
        return _colour_bomb_v3_layout(env, features, render_background, frame_env)
    if hasattr(env, "_grid_size"):
        return _matrix_grid_layout(env, features, render_background, frame_env)
    raise ValueError(
        f"{type(env).__name__} is not a supported custom MASA tabular "
        "environment for shield visualisation.",
    )


def _normalise_fixed_features(
    fixed_features: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if fixed_features is None:
        return {}
    if not isinstance(fixed_features, Mapping):
        raise TypeError("fixed_features must be a mapping from feature name to value.")
    return dict(fixed_features)


def _reject_unknown_features(
    env: Any,
    features: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(features) - allowed)
    if unknown:
        supported = ", ".join(sorted(allowed)) if allowed else "none"
        raise ValueError(
            f"{type(env).__name__} does not support fixed_features {unknown}; "
            f"supported features: {supported}.",
        )


def _frozen_lake_layout(
    env: CustomFrozenLake,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    _reject_unknown_features(env, features, set())
    nrow, ncol = int(env.nrow), int(env.ncol)
    cells = tuple(
        _Cell(
            state=row * ncol + col,
            row=row,
            col=col,
            description=f"at row={row}, col={col}",
        )
        for row in range(nrow)
        for col in range(ncol)
    )
    return _ShieldLayout(
        background=(
            _render_native_background(env, render_env) if render_background else None
        ),
        nrow=nrow,
        ncol=ncol,
        cells=cells,
        action_deltas=FROZEN_ACTION_DELTAS,
        action_names=FROZEN_ACTION_NAMES,
        title="FrozenLake shield",
    )


def _matrix_grid_layout(
    env: Any,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    _reject_unknown_features(env, features, set())
    size = int(env._grid_size)
    cells = tuple(
        _Cell(
            state=row * size + col,
            row=row,
            col=col,
            description=f"at row={row}, col={col}",
        )
        for row in range(size)
        for col in range(size)
    )
    return _ShieldLayout(
        background=(
            _render_native_background(env, render_env) if render_background else None
        ),
        nrow=size,
        ncol=size,
        cells=cells,
        action_deltas=_deltas_from_action_map(ACTION_MAP),
        action_names=GRID_ACTION_NAMES,
        title=f"{type(env).__name__} shield",
    )


def _colour_bomb_v3_layout(
    env: CustomColourBombGridWorldV3,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    zone, zone_name = _resolve_zone(env, features)
    size = int(env._grid_size)
    grid_area = size**2
    cells = tuple(
        _Cell(
            state=zone * grid_area + row * size + col,
            row=row,
            col=col,
            description=(
                f"at row={row}, col={col}, zone={zone}, "
                f"active_colour={zone_name}"
            ),
        )
        for row in range(size)
        for col in range(size)
    )
    # Render the frame with the active colour zone matching the slice, keeping the
    # start agent cell purely for display.
    render_state = zone * grid_area + (int(env._start_state) % grid_area)
    return _ShieldLayout(
        background=(
            _render_native_background(env, render_env, render_state=render_state)
            if render_background
            else None
        ),
        nrow=size,
        ncol=size,
        cells=cells,
        action_deltas=_deltas_from_action_map(ACTION_MAP),
        action_names=GRID_ACTION_NAMES,
        title=f"{type(env).__name__} shield, zone={zone_name}",
    )


def _resolve_zone(
    env: CustomColourBombGridWorldV3,
    features: Mapping[str, Any],
) -> tuple[int, str]:
    _reject_unknown_features(env, features, {"zone", "active_colour"})
    colour_by_zone = dict(getattr(env, "_active_colour_dict", {}))
    zone_by_colour = {str(colour): int(zone) for zone, colour in colour_by_zone.items()}

    zone = None
    if "zone" in features:
        zone = _int_feature(env, "zone", features["zone"])
    if "active_colour" in features:
        value = features["active_colour"]
        if isinstance(value, str):
            if value not in zone_by_colour:
                raise ValueError(
                    f"active_colour must be one of {sorted(zone_by_colour)}, "
                    f"got {value!r}.",
                )
            colour_zone = zone_by_colour[value]
        else:
            colour_zone = _int_feature(env, "active_colour", value)
        if zone is not None and colour_zone != zone:
            raise ValueError(
                "fixed_features contains conflicting zone and active_colour "
                f"values: {zone} and {colour_zone}.",
            )
        zone = colour_zone
    if zone is None:
        zone = 0

    if zone < 0 or zone >= int(env._n_coloured_zones):
        raise ValueError(
            f"zone must be in [0, {int(env._n_coloured_zones) - 1}], got {zone}.",
        )
    return zone, str(colour_by_zone.get(zone, zone))


def _pacman_layout(
    env: Any,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    fixed = _resolve_pacman_features(env, features)
    cells: list[_Cell] = []
    for row in range(int(env._n_row)):
        for col in range(int(env._n_col)):
            key = (
                row,
                col,
                fixed["agent_direction"],
                fixed["ghost_y"],
                fixed["ghost_x"],
                fixed["ghost_direction"],
                fixed["food"],
            )
            if key not in env._state_map:
                continue
            cells.append(
                _Cell(
                    state=int(env._state_map[key]),
                    row=row,
                    col=col,
                    description=(
                        f"at agent_y={row}, agent_x={col}, "
                        f"agent_direction={fixed['agent_direction']}, "
                        f"ghost_y={fixed['ghost_y']}, "
                        f"ghost_x={fixed['ghost_x']}, "
                        f"ghost_direction={fixed['ghost_direction']}, "
                        f"food={fixed['food']}"
                    ),
                ),
            )
    if not cells:
        raise ValueError("fixed_features does not leave any valid Pacman states.")
    # Render the frame with the ghost (and other fixed features) at the requested
    # slice, keeping the start agent cell purely for display.
    start = env._reverse_state_map[int(env._start_state)]
    render_key = (
        start[0],
        start[1],
        fixed["agent_direction"],
        fixed["ghost_y"],
        fixed["ghost_x"],
        fixed["ghost_direction"],
        fixed["food"],
    )
    render_state = (
        int(env._state_map[render_key])
        if render_key in env._state_map
        else int(cells[0].state)
    )
    return _ShieldLayout(
        background=(
            _render_native_background(env, render_env, render_state=render_state)
            if render_background
            else None
        ),
        nrow=int(env._n_row),
        ncol=int(env._n_col),
        cells=tuple(cells),
        action_deltas=_deltas_from_action_map(PACMAN_ACTION_MAP),
        action_names=PACMAN_ACTION_NAMES,
        title=f"{type(env).__name__} shield",
    )


def _resolve_pacman_features(env: Any, features: Mapping[str, Any]) -> dict[str, int]:
    allowed = {
        "agent_direction",
        "ghost_y",
        "ghost_x",
        "ghost_direction",
        "food",
    }
    _reject_unknown_features(env, features, allowed)
    start = env._reverse_state_map[int(env._start_state)]
    fixed = {
        "agent_direction": int(start[2]),
        "ghost_y": int(start[3]),
        "ghost_x": int(start[4]),
        "ghost_direction": int(start[5]),
        "food": int(start[6]),
    }
    for name, value in features.items():
        fixed[name] = _int_feature(env, name, value)

    _validate_range("agent_direction", fixed["agent_direction"], 0, 4)
    _validate_range("ghost_direction", fixed["ghost_direction"], 0, 4)
    _validate_range("food", fixed["food"], 0, 2)
    _validate_range("ghost_y", fixed["ghost_y"], 0, int(env._n_row))
    _validate_range("ghost_x", fixed["ghost_x"], 0, int(env._n_col))
    if int(env._layout[fixed["ghost_y"], fixed["ghost_x"]]) != 0:
        raise ValueError(
            "ghost_y and ghost_x must identify a free Pacman cell, got "
            f"({fixed['ghost_y']}, {fixed['ghost_x']}).",
        )
    return fixed


def _media_v0_layout(
    env: CustomMediaStreaming,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    _reject_unknown_features(env, features, set())
    states = list(range(int(env._buffer_size)))
    return _line_layout(
        env,
        states,
        [f"buffer={state}" for state in states],
        title=f"{type(env).__name__} shield",
        background=(
            _render_native_background(env, render_env) if render_background else None
        ),
    )


def _media_v2_layout(
    env: CustomMediaStreamingV2,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    _reject_unknown_features(env, features, {"fast_count"})
    _buffer, start_fast_count = env._decode_state(int(env._start_state))
    fast_count = _int_feature(
        env,
        "fast_count",
        features.get("fast_count", start_fast_count),
    )
    _validate_range("fast_count", fast_count, 0, int(env._fast_count_cap) + 1)
    states = [
        int(env._encode_state(buffer_level, fast_count))
        for buffer_level in range(int(env._buffer_size))
    ]
    return _line_layout(
        env,
        states,
        [
            f"buffer={buffer_level}, fast_count={fast_count}"
            for buffer_level in range(int(env._buffer_size))
        ],
        title=f"{type(env).__name__} shield, fast_count={fast_count}",
        background=(
            _render_native_background(env, render_env) if render_background else None
        ),
    )


def _media_v3_layout(
    env: CustomMediaStreamingV3,
    features: Mapping[str, Any],
    render_background: bool,
    render_env: Any,
) -> _ShieldLayout:
    _reject_unknown_features(env, features, {"time"})
    _danger, start_time = env._decode_safety_state(int(env._start_state))
    time = _int_feature(env, "time", features.get("time", start_time))
    _validate_range("time", time, 0, int(env._time_states))
    states = [
        int(env._encode_safety_state(danger_level, time))
        for danger_level in range(int(env._danger_states))
    ]
    return _line_layout(
        env,
        states,
        [
            f"danger={danger_level}, time={time}"
            for danger_level in range(int(env._danger_states))
        ],
        title=f"{type(env).__name__} shield, time={time}",
        background=(
            _render_native_background(env, render_env) if render_background else None
        ),
    )


def _line_layout(
    env: Any,
    states: list[int],
    descriptions: list[str],
    *,
    title: str,
    background: np.ndarray | None,
) -> _ShieldLayout:
    ncol = len(states)
    centers = _media_axis_centers(background, ncol)
    cells = []
    for idx, state in enumerate(states):
        center = centers[idx] if centers is not None else None
        kwargs = {}
        if center is not None:
            kwargs = {
                "center_x": center[0],
                "center_y": center[1],
                "span_w": center[2],
                "span_h": center[3],
            }
        cells.append(
            _Cell(
                state=state,
                row=0,
                col=idx,
                description=descriptions[idx],
                **kwargs,
            ),
        )
    return _ShieldLayout(
        background=background,
        nrow=1,
        ncol=ncol,
        cells=tuple(cells),
        action_deltas=MEDIA_ACTION_DELTAS,
        action_names=MEDIA_ACTION_NAMES,
        title=title,
    )


def _int_feature(env: Any, name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{type(env).__name__} fixed feature {name!r} must be an integer, "
            f"got {value!r}.",
        ) from exc


def _validate_range(name: str, value: int, lower: int, upper: int) -> None:
    if value < lower or value >= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper - 1}], got {value}.")


def _deltas_from_action_map(
    action_map: Mapping[int, tuple[int, int]],
) -> dict[int, tuple[float, float]]:
    deltas = {}
    for action, (dy, dx) in action_map.items():
        deltas[int(action)] = (0.30 * float(dx), 0.30 * float(dy))
    return deltas


def _render_native_background(
    env: Any,
    render_env: Any,
    *,
    render_state: int | None = None,
) -> np.ndarray:
    if not hasattr(render_env, "render"):
        raise ValueError(f"{type(render_env).__name__} does not expose render().")

    # Render via the unwrapped env so Gymnasium's OrderEnforcer wrapper (which
    # requires reset() before render()) doesn't apply; calling reset() here
    # would mutate caller-visible env state, which plot_tabular_shield must not do.
    unwrapped_render_env = getattr(render_env, "unwrapped", render_env)

    # ``render_state`` lets a slice render the background from a state whose
    # non-coordinate features (e.g. Pacman's ghost, the colour-bomb active zone)
    # match the requested ``fixed_features``, since the renderers decode those
    # features straight from ``env._state``. The original state is restored below
    # so the call stays side-effect-free.
    set_state = render_state is not None and hasattr(env, "_state")
    old_state = getattr(env, "_state", None)
    old_render_mode = getattr(env, "render_mode", None)
    try:
        if set_state:
            env._state = int(render_state)
        if old_render_mode != "rgb_array":
            env.render_mode = "rgb_array"
        frame = unwrapped_render_env.render()
    finally:
        if set_state:
            env._state = old_state
        if hasattr(env, "render_mode"):
            env.render_mode = old_render_mode

    if frame is None:
        raise ValueError(
            f"{type(render_env).__name__}.render() did not return an "
            "rgb_array frame.",
        )

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"{type(render_env).__name__}.render() must return an RGB array with "
            f"shape (height, width, 3), got {frame.shape}.",
        )
    return np.array(frame, copy=True)


def _media_axis_centers(
    background: np.ndarray | None,
    n_positions: int,
) -> list[tuple[float, float, float, float]] | None:
    if background is None:
        return None

    height, width = background.shape[:2]
    scale = 3
    high_width = int(width * scale)
    high_height = int(height * scale)
    margin = max(18, high_width // 18)
    panel_left = margin
    panel_right = high_width - margin
    gauge_left = panel_left + margin // 2
    gauge_right = panel_right - margin // 2
    gauge_top = margin + int(high_height * 0.38)
    gauge_height = max(22, high_height // 8)

    axis_left = gauge_left / scale
    axis_right = gauge_right / scale
    axis_top = gauge_top / scale
    axis_height = gauge_height / scale
    span_w = (axis_right - axis_left) / max(1, n_positions)
    center_y = axis_top + axis_height / 2.0
    return [
        (axis_left + (idx + 0.5) * span_w, center_y, span_w, axis_height)
        for idx in range(n_positions)
    ]


def _labels_for_state(env: Any, state: int) -> set[str]:
    if not hasattr(env, "label_fn"):
        return set()
    try:
        return set(env.label_fn(int(state)))
    except Exception:
        return set()


def _action_risk(info: Any | None, shield_shape: tuple[int, int]) -> np.ndarray | None:
    if info is None or getattr(info, "action_risk", None) is None:
        return None
    action_risk = np.asarray(info.action_risk, dtype=float)
    if action_risk.shape != shield_shape:
        raise ValueError(
            f"info.action_risk must have shape {shield_shape}, "
            f"got {action_risk.shape}.",
        )
    return action_risk


def _action_color(plt: Any, risk: float | None) -> Any:
    if risk is None:
        return "#0f766e"
    return plt.cm.RdYlGn_r(float(np.clip(risk, 0.0, 1.0)))


def _draw_action(
    ax: Any,
    cell: _Cell,
    action: int,
    layout: _ShieldLayout,
    cell_w: float,
    cell_h: float,
    color: Any,
    risk: float | None,
    show_risk: bool,
) -> None:
    cx = (
        float(cell.center_x)
        if cell.center_x is not None
        else (cell.col + 0.5) * cell_w
    )
    cy = (
        float(cell.center_y)
        if cell.center_y is not None
        else (cell.row + 0.5) * cell_h
    )
    span_w = float(cell.span_w) if cell.span_w is not None else cell_w
    span_h = float(cell.span_h) if cell.span_h is not None else cell_h
    dx, dy = layout.action_deltas.get(action, (0.0, 0.0))

    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        ax.plot(cx, cy, marker="o", markersize=5, color=color, alpha=0.95)
    else:
        ax.annotate(
            "",
            xy=(cx + dx * span_w, cy + dy * span_h),
            xytext=(cx, cy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.2,
                "alpha": 0.95,
            },
        )

    if show_risk and risk is not None and risk > 0.0:
        text_x = cx + (dx if abs(dx) > 1e-12 else 0.0) * span_w * 1.18
        text_y = cy + (dy if abs(dy) > 1e-12 else -0.22) * span_h * 1.18
        ax.text(
            text_x,
            text_y,
            f"{risk:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.12",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )


def _figure_size(layout: _ShieldLayout) -> tuple[float, float]:
    if layout.background is None:
        ratio = layout.ncol / max(1.0, float(layout.nrow))
    else:
        height_px, width_px = layout.background.shape[:2]
        ratio = width_px / max(1.0, float(height_px))
    width = float(np.clip(4.5 * ratio, 4.5, 12.0))
    height = float(np.clip(width / max(1.0, ratio), 2.5, 8.0))
    return width, height


__all__ = [
    "plot_tabular_shield",
    "print_allowed_actions",
    "render_tabular_shield_background",
]
