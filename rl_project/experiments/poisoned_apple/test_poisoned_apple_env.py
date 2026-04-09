"""Tests for PoisonedAppleEnv Gymnasium compliance and behavior."""

from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rl_project.experiments.poisoned_apple.poisoned_apple_env import (
    PoisonedAppleEnv,
    get_safety_critical_observations_and_safe_actions,
    plot_safety_dataset_on_grid,
)


@pytest.mark.parametrize("bad_render_mode", ["ansi", "rgb", "text"])
def test_constructor_rejects_invalid_render_mode(bad_render_mode: str) -> None:
    with pytest.raises(ValueError, match="Invalid render_mode"):
        PoisonedAppleEnv(render_mode=bad_render_mode)


def test_constructor_rejects_overlapping_fixed_positions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PoisonedAppleEnv(
            num_apples=2,
            num_poisoned=1,
            safe_apple_positions=[(0, 0)],
            poisoned_apple_positions=[(0, 0)],
        )


def test_constructor_rejects_out_of_bounds_fixed_position() -> None:
    with pytest.raises(ValueError, match="out-of-bounds"):
        PoisonedAppleEnv(
            grid_size=4,
            num_apples=1,
            num_poisoned=0,
            safe_apple_positions=[(4, 0)],
        )


def test_partial_fixed_positions_fill_remaining_randomly() -> None:
    env = PoisonedAppleEnv(
        grid_size=5,
        num_apples=5,
        num_poisoned=2,
        agent_start_pos=(4, 4),
        safe_apple_positions=[(0, 1)],
        poisoned_apple_positions=[(1, 1)],
        observation_type="flat",
    )
    env.reset(seed=7)

    assert tuple(env.agent_pos) == (4, 4)
    assert (0, 1) in env.safe_apples
    assert (1, 1) in env.poisoned_apples
    assert len(env.safe_apples) == 3
    assert len(env.poisoned_apples) == 2
    assert tuple(env.agent_pos) not in env.safe_apples
    assert tuple(env.agent_pos) not in env.poisoned_apples

    env.close()


def test_reset_seed_is_deterministic() -> None:
    env = PoisonedAppleEnv(grid_size=6, num_apples=4, num_poisoned=1, observation_type="flat")

    obs1, _ = env.reset(seed=123)
    state1 = (
        tuple(env.agent_pos),
        tuple(sorted(env.safe_apples)),
        tuple(sorted(env.poisoned_apples)),
        obs1.copy(),
    )

    obs2, _ = env.reset(seed=123)
    state2 = (
        tuple(env.agent_pos),
        tuple(sorted(env.safe_apples)),
        tuple(sorted(env.poisoned_apples)),
        obs2.copy(),
    )

    assert state1[0] == state2[0]
    assert state1[1] == state2[1]
    assert state1[2] == state2[2]
    assert np.array_equal(state1[3], state2[3])

    env.close()


@pytest.mark.parametrize("observation_type", ["flat", "image", "coordinates"])
def test_step_contract_and_observation_space(observation_type: str) -> None:
    env = PoisonedAppleEnv(
        grid_size=5,
        num_apples=3,
        num_poisoned=1,
        observation_type=observation_type,
    )

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    obs2, reward, terminated, truncated, info2 = env.step(PoisonedAppleEnv.UP)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)

    env.close()


def test_invalid_action_raises_error() -> None:
    env = PoisonedAppleEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(999)
    env.close()


def test_step_before_reset_raises_runtime_error() -> None:
    env = PoisonedAppleEnv()
    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(PoisonedAppleEnv.UP)
    env.close()


def test_step_after_episode_end_requires_reset() -> None:
    env = PoisonedAppleEnv(
        grid_size=3,
        num_apples=1,
        num_poisoned=0,
        agent_start_pos=(1, 1),
        safe_apple_positions=[(1, 2)],
        poisoned_apple_positions=[],
    )
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(PoisonedAppleEnv.RIGHT)
    assert terminated is True
    assert truncated is False

    with pytest.raises(RuntimeError, match="Episode has ended"):
        env.step(PoisonedAppleEnv.RIGHT)

    env.close()


def test_safe_only_termination_keeps_poisoned_remaining() -> None:
    env = PoisonedAppleEnv(
        grid_size=3,
        num_apples=2,
        num_poisoned=1,
        agent_start_pos=(1, 1),
        safe_apple_positions=[(1, 2)],
        poisoned_apple_positions=[(1, 0)],
    )
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step(PoisonedAppleEnv.RIGHT)
    assert terminated is True
    assert truncated is False
    assert info["safe_apples_remaining"] == 0
    assert info["poisoned_apples_remaining"] == 1

    env.close()


def test_truncation_when_max_steps_reached() -> None:
    env = PoisonedAppleEnv(
        grid_size=3,
        num_apples=1,
        num_poisoned=0,
        max_steps=1,
        agent_start_pos=(2, 2),
        safe_apple_positions=[(0, 0)],
        poisoned_apple_positions=[],
    )
    env.reset(seed=0)

    _, _, terminated, truncated, _ = env.step(PoisonedAppleEnv.RIGHT)
    assert terminated is False
    assert truncated is True

    env.close()


def test_safety_info_flags_and_cost() -> None:
    env = PoisonedAppleEnv(
        grid_size=3,
        num_apples=2,
        num_poisoned=1,
        agent_start_pos=(1, 1),
        safe_apple_positions=[(0, 0)],
        poisoned_apple_positions=[(1, 2)],
    )

    _, info0 = env.reset(seed=0)
    assert info0["safe"] is True
    assert info0["cost"] == 0.0
    assert info0["ate_safe_apple"] is False
    assert info0["ate_poisoned_apple"] is False

    _, _, _, _, info1 = env.step(PoisonedAppleEnv.RIGHT)
    assert info1["safe"] is False
    assert info1["cost"] == 1.0
    assert info1["ate_safe_apple"] is False
    assert info1["ate_poisoned_apple"] is True

    _, _, _, _, info2 = env.step(PoisonedAppleEnv.LEFT)
    assert info2["safe"] is True
    assert info2["cost"] == 0.0
    assert info2["ate_safe_apple"] is False
    assert info2["ate_poisoned_apple"] is False

    env.close()


def test_render_rgb_array_shape_and_dtype() -> None:
    env = PoisonedAppleEnv(grid_size=4, render_mode="rgb_array")
    env.reset(seed=0)

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0

    env.close()


def test_render_with_none_mode_raises_error() -> None:
    env = PoisonedAppleEnv(render_mode=None)
    env.reset(seed=0)

    with pytest.raises(ValueError, match="render_mode is None"):
        env.render()

    env.close()


def test_human_render_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    env = PoisonedAppleEnv(render_mode="human")
    env.reset(seed=0)
    out = env.render()
    assert out is None
    captured = capsys.readouterr()
    assert "Legend" not in captured.out
    assert captured.out == ""
    env.close()


def test_repeated_render_calls_do_not_crash() -> None:
    env_rgb = PoisonedAppleEnv(grid_size=5, render_mode="rgb_array")
    env_rgb.reset(seed=0)
    for _ in range(3):
        frame = env_rgb.render()
        assert isinstance(frame, np.ndarray)
        _, _, terminated, truncated, _ = env_rgb.step(PoisonedAppleEnv.RIGHT)
        if terminated or truncated:
            break
    env_rgb.close()

    env_human = PoisonedAppleEnv(grid_size=5, render_mode="human")
    env_human.reset(seed=0)
    for _ in range(3):
        out = env_human.render()
        assert out is None
        _, _, terminated, truncated, _ = env_human.step(PoisonedAppleEnv.RIGHT)
        if terminated or truncated:
            break
    env_human.close()


def test_rgb_render_shape_is_stable_across_calls() -> None:
    env = PoisonedAppleEnv(grid_size=4, render_mode="rgb_array")
    env.reset(seed=0)
    frame0 = env.render()
    frame1 = env.render()
    assert frame0.shape == frame1.shape
    env.close()


def test_module_import_without_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "rl_project.experiments.poisoned_apple.poisoned_apple_env"
    original_import = builtins.__import__

    def no_matplotlib_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib_import)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    assert hasattr(module, "PoisonedAppleEnv")


def test_critical_safe_actions_requires_reset() -> None:
    env = PoisonedAppleEnv(observation_type="flat")
    with pytest.raises(ValueError, match="must be reset"):
        get_safety_critical_observations_and_safe_actions(env)
    env.close()


def test_critical_safe_actions_rejects_non_flat_obs_type() -> None:
    env = PoisonedAppleEnv(observation_type="image")
    env.reset(seed=0)
    with pytest.raises(NotImplementedError, match="only observation_type='flat'"):
        get_safety_critical_observations_and_safe_actions(env)
    env.close()


def test_critical_safe_actions_known_layout() -> None:
    env = PoisonedAppleEnv(
        grid_size=3,
        num_apples=1,
        num_poisoned=1,
        safe_apple_positions=[],
        poisoned_apple_positions=[(1, 1)],
        observation_type="flat",
    )
    env.reset(seed=0)
    critical = get_safety_critical_observations_and_safe_actions(env)

    expected = {
        (0, 1): [PoisonedAppleEnv.UP, PoisonedAppleEnv.RIGHT, PoisonedAppleEnv.LEFT],
        (1, 0): [PoisonedAppleEnv.UP, PoisonedAppleEnv.DOWN, PoisonedAppleEnv.LEFT],
        (1, 2): [PoisonedAppleEnv.UP, PoisonedAppleEnv.RIGHT, PoisonedAppleEnv.DOWN],
        (2, 1): [PoisonedAppleEnv.RIGHT, PoisonedAppleEnv.DOWN, PoisonedAppleEnv.LEFT],
    }
    assert len(critical) == len(expected)

    # Deterministic row-major order.
    ordered_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for (obs, safe_actions), agent_pos in zip(critical, ordered_positions):
        assert obs.shape == (env.grid_size * env.grid_size,)
        assert obs.dtype == np.float32
        idx_agent = int(np.flatnonzero(obs == 1.0)[0])
        decoded = (idx_agent // env.grid_size, idx_agent % env.grid_size)
        assert decoded == agent_pos
        assert safe_actions == expected[agent_pos]

    env.close()


def test_critical_safe_actions_deterministic_for_same_layout() -> None:
    env = PoisonedAppleEnv(
        grid_size=4,
        num_apples=2,
        num_poisoned=1,
        safe_apple_positions=[(0, 0)],
        poisoned_apple_positions=[(2, 2)],
        observation_type="flat",
    )
    env.reset(seed=11)
    out1 = get_safety_critical_observations_and_safe_actions(env)
    out2 = get_safety_critical_observations_and_safe_actions(env)

    assert len(out1) == len(out2)
    for (obs1, safe1), (obs2, safe2) in zip(out1, out2):
        assert np.array_equal(obs1, obs2)
        assert safe1 == safe2

    env.close()


def test_critical_safe_actions_empty_when_no_poisoned_apples() -> None:
    env = PoisonedAppleEnv(
        grid_size=4,
        num_apples=1,
        num_poisoned=0,
        safe_apple_positions=[(1, 1)],
        poisoned_apple_positions=[],
        observation_type="flat",
    )
    env.reset(seed=0)
    critical = get_safety_critical_observations_and_safe_actions(env)
    assert critical == []
    env.close()


def test_plot_safety_dataset_on_grid_from_list_runs() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")

    env = PoisonedAppleEnv(
        grid_size=4,
        num_apples=2,
        num_poisoned=1,
        safe_apple_positions=[(0, 0)],
        poisoned_apple_positions=[(2, 2)],
        observation_type="flat",
    )
    env.reset(seed=0)
    critical = get_safety_critical_observations_and_safe_actions(env)

    fig = plot_safety_dataset_on_grid(env=env, safety_dataset=critical, title="Safety")
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1
    plt.close(fig)
    env.close()


def test_plot_safety_dataset_on_grid_from_tensordataset_runs() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    torch = pytest.importorskip("torch")

    env = PoisonedAppleEnv(
        grid_size=4,
        num_apples=2,
        num_poisoned=1,
        safe_apple_positions=[(0, 0)],
        poisoned_apple_positions=[(2, 2)],
        observation_type="flat",
    )
    env.reset(seed=0)
    critical = get_safety_critical_observations_and_safe_actions(env)

    X = torch.tensor(np.stack([obs for obs, _ in critical]), dtype=torch.float32)
    Y = torch.zeros((len(critical), env.action_space.n), dtype=torch.float32)
    for i, (_, safe_actions) in enumerate(critical):
        Y[i, safe_actions] = 1.0
    ds = torch.utils.data.TensorDataset(X, Y)

    fig = plot_safety_dataset_on_grid(env=env, safety_dataset=ds, title="Safety DS")
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1
    plt.close(fig)
    env.close()
