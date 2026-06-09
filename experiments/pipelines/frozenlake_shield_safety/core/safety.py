"""Safety dataset, rollout, and supervised fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset

from experiments.pipelines.frozenlake_shield_safety.core.env import ACTION_DELTAS, grid_shape, state_index_to_obs
from experiments.utils.shield_utils import ShieldSynthesisInfo, ShieldType, synthesise_shield


RashomonPayload = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    state_index: int
    row: int
    col: int
    action: int


@dataclass(frozen=True)
class RolloutResult:
    steps: tuple[TrajectoryStep, ...]
    total_reward: float
    failed: bool
    terminated: bool
    truncated: bool

    @property
    def failure_rate(self) -> float:
        return 1.0 if self.failed else 0.0

    def state_action_pairs(self) -> list[dict[str, int]]:
        return [
            {
                "step": step.step,
                "state_index": step.state_index,
                "row": step.row,
                "col": step.col,
                "action": step.action,
            }
            for step in self.steps
        ]


def _cell(env_map: list[str] | tuple[str, ...], state_index: int) -> str:
    _, ncol = grid_shape(env_map)
    return env_map[state_index // ncol][state_index % ncol]


def _next_state_index(
    env_map: list[str] | tuple[str, ...],
    state_index: int,
    action: int,
) -> int:
    nrow, ncol = grid_shape(env_map)
    row = state_index // ncol
    col = state_index % ncol
    dr, dc = ACTION_DELTAS[action]
    next_row = min(max(row + dr, 0), nrow - 1)
    next_col = min(max(col + dc, 0), ncol - 1)
    return next_row * ncol + next_col


def frozenlake_transition_matrix(env_map: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return deterministic FrozenLake dynamics as P[next_state, state, action]."""
    nrow, ncol = grid_shape(env_map)
    n_states = nrow * ncol
    matrix = np.zeros((n_states, n_states, 4), dtype=np.float64)

    for state in range(n_states):
        cell = _cell(env_map, state)
        for action in range(4):
            next_state = state if cell in {"H", "G"} else _next_state_index(env_map, state, action)
            matrix[next_state, state, action] = 1.0
    return matrix


def frozenlake_label_fn(env_map: list[str] | tuple[str, ...], state: int) -> set[str]:
    """Label holes as unsafe for shield synthesis."""
    return {"unsafe"} if _cell(env_map, int(state)) == "H" else set()


def frozenlake_cost_fn(labels: set[str]) -> float:
    return 1.0 if "unsafe" in labels else 0.0


def synthesise_frozenlake_shield(
    env_map: list[str] | tuple[str, ...],
    *,
    shield_type: ShieldType = "deterministic",
    risk_threshold: float = 0.0,
    theta: float = 1e-10,
    max_vi_steps: int = 1000,
    unsafe_cost_threshold: float = 0.5,
) -> tuple[np.ndarray, ShieldSynthesisInfo]:
    """Synthesise a tabular shield for a non-slippery FrozenLake map."""
    env = gym.make("FrozenLake-v1", desc=list(env_map), is_slippery=False)
    try:
        shield, info = synthesise_shield(
            env,
            lambda _env: frozenlake_transition_matrix(env_map),
            lambda state: frozenlake_label_fn(env_map, int(state)),
            frozenlake_cost_fn,
            shield_type=shield_type,
            risk_threshold=risk_threshold,
            theta=theta,
            max_vi_steps=max_vi_steps,
            unsafe_cost_threshold=unsafe_cost_threshold,
            use_masa_helper=False,
            return_info=True,
        )
    finally:
        env.close()
    return np.asarray(shield, dtype=np.int64), info


def create_shield_rashomon_dataset(
    env_map: list[str] | tuple[str, ...],
    *,
    task_num: float,
    shield: np.ndarray,
) -> RashomonPayload:
    """Convert synthesized shield rows into multi-hot Rashomon targets."""
    nrow, ncol = grid_shape(env_map)
    expected_shape = (nrow * ncol, 4)
    shield_arr = np.asarray(shield)
    if shield_arr.shape != expected_shape:
        raise ValueError(f"Expected shield shape {expected_shape}, got {shield_arr.shape}.")

    states: list[int] = []
    action_masks: list[np.ndarray] = []
    for state_index in traversable_nonterminal_states(env_map):
        mask = (shield_arr[state_index] > 0).astype(np.float32)
        if float(mask.sum()) <= 0.0:
            continue
        states.append(state_index)
        action_masks.append(mask)

    if not states:
        return {
            "state": torch.empty((0, 3), dtype=torch.float32),
            "actions": torch.empty((0, 4), dtype=torch.float32),
        }

    state_tensors = [
        state_index_to_obs(state_index, env_map, task_num)
        for state_index in states
    ]
    return {
        "state": torch.tensor(np.asarray(state_tensors), dtype=torch.float32),
        "actions": torch.tensor(np.asarray(action_masks), dtype=torch.float32),
    }


def min_risk_shield_from_action_risk(
    env_map: list[str] | tuple[str, ...],
    action_risk: np.ndarray,
    *,
    theta: float,
) -> np.ndarray:
    """Return a mask allowing minimum-risk actions in each trainable state."""
    nrow, ncol = grid_shape(env_map)
    expected_shape = (nrow * ncol, 4)
    risk_arr = np.asarray(action_risk, dtype=np.float64)
    if risk_arr.shape != expected_shape:
        raise ValueError(f"Expected action_risk shape {expected_shape}, got {risk_arr.shape}.")
    if theta < 0.0:
        raise ValueError(f"theta must be non-negative for min-risk masks, got {theta}.")

    shield = np.zeros(expected_shape, dtype=np.int64)
    for state_index in traversable_nonterminal_states(env_map):
        state_risk = risk_arr[state_index]
        min_risk = float(np.min(state_risk))
        shield[state_index] = (state_risk <= min_risk + float(theta)).astype(np.int64)
    return shield


def shield_allowed_action_risk_stats(
    env_map: list[str] | tuple[str, ...],
    shield: np.ndarray,
    action_risk: np.ndarray | None,
    *,
    prefix: str = "dataset_allowed_action_risk",
) -> dict[str, float | int]:
    """Summarise action risks for allowed trainable-state actions."""
    if action_risk is None:
        return {}
    nrow, ncol = grid_shape(env_map)
    expected_shape = (nrow * ncol, 4)
    shield_arr = np.asarray(shield)
    risk_arr = np.asarray(action_risk, dtype=np.float64)
    if shield_arr.shape != expected_shape:
        raise ValueError(f"Expected shield shape {expected_shape}, got {shield_arr.shape}.")
    if risk_arr.shape != expected_shape:
        raise ValueError(f"Expected action_risk shape {expected_shape}, got {risk_arr.shape}.")

    risks: list[float] = []
    for state_index in traversable_nonterminal_states(env_map):
        for action in np.flatnonzero(shield_arr[state_index] > 0):
            risks.append(float(risk_arr[state_index, int(action)]))

    if not risks:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
        }
    risk_array = np.asarray(risks, dtype=np.float64)
    return {
        f"{prefix}_count": int(risk_array.size),
        f"{prefix}_min": float(risk_array.min()),
        f"{prefix}_max": float(risk_array.max()),
        f"{prefix}_mean": float(risk_array.mean()),
    }


def shield_allowed_action_stats(
    env_map: list[str] | tuple[str, ...],
    shield: np.ndarray,
    payload: RashomonPayload,
) -> dict[str, int | float]:
    """Return YAML-friendly allowed-action summary statistics."""
    shield_arr = np.asarray(shield)
    all_counts = shield_arr.sum(axis=1).astype(np.float64)
    train_counts = payload["actions"].sum(dim=1).cpu().numpy().astype(np.float64)
    return {
        "shield_state_count": int(shield_arr.shape[0]),
        "shield_allowed_action_count": int(shield_arr.sum()),
        "shield_allowed_actions_min": float(all_counts.min()) if all_counts.size else 0.0,
        "shield_allowed_actions_max": float(all_counts.max()) if all_counts.size else 0.0,
        "shield_allowed_actions_mean": float(all_counts.mean()) if all_counts.size else 0.0,
        "shield_trainable_state_count": int(payload["state"].shape[0]),
        "shield_trainable_allowed_actions_min": float(train_counts.min()) if train_counts.size else 0.0,
        "shield_trainable_allowed_actions_max": float(train_counts.max()) if train_counts.size else 0.0,
        "shield_trainable_allowed_actions_mean": float(train_counts.mean()) if train_counts.size else 0.0,
        "shield_traversable_nonterminal_state_count": int(len(traversable_nonterminal_states(env_map))),
    }


def safe_action_mask_for_state(
    env_map: list[str] | tuple[str, ...],
    state_index: int,
) -> np.ndarray:
    """Return a multi-hot mask where deterministic actions avoiding holes are safe."""
    cell = _cell(env_map, state_index)
    if cell in {"H", "G"}:
        raise ValueError(f"State {state_index} is terminal/non-traversable ({cell}).")
    mask = np.zeros(4, dtype=np.float32)
    for action in range(4):
        next_state = _next_state_index(env_map, state_index, action)
        mask[action] = 0.0 if _cell(env_map, next_state) == "H" else 1.0
    return mask


def traversable_nonterminal_states(env_map: list[str] | tuple[str, ...]) -> list[int]:
    nrow, ncol = grid_shape(env_map)
    states: list[int] = []
    for state in range(nrow * ncol):
        if _cell(env_map, state) in {"S", "F"}:
            states.append(state)
    return states


def create_rashomon_dataset(
    env_map: list[str] | tuple[str, ...],
    *,
    task_num: float,
) -> RashomonPayload:
    """Enumerate source states with synthesized deterministic shield actions."""
    shield, _ = synthesise_frozenlake_shield(env_map, shield_type="deterministic")
    return create_shield_rashomon_dataset(env_map, task_num=task_num, shield=shield)


def validate_rashomon_payload(payload: RashomonPayload) -> None:
    if set(payload.keys()) != {"state", "actions"}:
        raise ValueError(f"Expected Rashomon payload keys {{'state', 'actions'}}, got {sorted(payload.keys())}.")
    state = payload["state"]
    actions = payload["actions"]
    if not isinstance(state, torch.Tensor) or not isinstance(actions, torch.Tensor):
        raise TypeError("Rashomon payload values must be torch tensors.")
    if state.dtype != torch.float32 or actions.dtype != torch.float32:
        raise TypeError("Rashomon payload tensors must be float32.")
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(f"Expected state tensor shape [N, 3], got {tuple(state.shape)}.")
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected actions tensor shape [N, 4], got {tuple(actions.shape)}.")
    if state.shape[0] != actions.shape[0]:
        raise ValueError("Rashomon state/actions tensors must have the same first dimension.")
    if not torch.all((actions == 0.0) | (actions == 1.0)):
        raise ValueError("Rashomon action tensor must be multi-hot with 0/1 entries.")


def to_tensor_dataset(payload: RashomonPayload) -> TensorDataset:
    validate_rashomon_payload(payload)
    return TensorDataset(payload["state"], payload["actions"])


def greedy_action(
    actor: torch.nn.Module,
    obs: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> int:
    actor.eval()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(actor(obs_t).argmax(dim=1).item())


def rollout_greedy_policy(
    actor: torch.nn.Module,
    env: gym.Env,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> RolloutResult:
    actor.eval()
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    failed = False
    steps: list[TrajectoryStep] = []
    terminated = False
    truncated = False

    _, ncol = env.unwrapped.desc.shape
    step_idx = 0
    while not done:
        state_index = int(env.unwrapped.s)
        row = state_index // ncol
        col = state_index % ncol
        action = greedy_action(actor, obs, device=device)
        steps.append(
            TrajectoryStep(
                step=step_idx,
                state_index=state_index,
                row=row,
                col=col,
                action=action,
            ),
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        failed = failed or (not bool(info.get("safe", True)))
        done = bool(terminated or truncated)
        step_idx += 1

    return RolloutResult(
        steps=tuple(steps),
        total_reward=total_reward,
        failed=failed,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def trajectory_action_map(steps: Iterable[TrajectoryStep]) -> dict[int, int]:
    action_by_state: dict[int, int] = {}
    for step in steps:
        previous = action_by_state.get(step.state_index)
        if previous is not None and previous != step.action:
            raise ValueError(
                f"Trajectory visits state {step.state_index} with conflicting actions: "
                f"{previous} and {step.action}.",
            )
        action_by_state[step.state_index] = step.action
    return action_by_state


def build_noadapt_supervised_payload(
    rashomon_payload: RashomonPayload,
    *,
    env_map: list[str] | tuple[str, ...],
    trajectory_steps: Iterable[TrajectoryStep],
) -> RashomonPayload:
    """Restrict trajectory states to their learned action; keep off-trajectory safety masks."""
    validate_rashomon_payload(rashomon_payload)
    states = rashomon_payload["state"].clone()
    actions = rashomon_payload["actions"].clone()
    action_by_state = trajectory_action_map(trajectory_steps)
    _, ncol = grid_shape(env_map)

    for row_idx, obs in enumerate(states):
        row = int(round(float(obs[0].item()) * (len(env_map) - 1)))
        col = int(round(float(obs[1].item()) * (ncol - 1)))
        state_index = row * ncol + col
        if state_index not in action_by_state:
            continue
        action = action_by_state[state_index]
        if actions[row_idx, action].item() <= 0:
            raise ValueError(f"Trajectory action {action} in state {state_index} is unsafe.")
        actions[row_idx].zero_()
        actions[row_idx, action] = 1.0

    return {
        "state": states,
        "actions": actions,
    }


def allowed_action_accuracy(
    actor: torch.nn.Module,
    payload: RashomonPayload,
    *,
    device: str | torch.device = "cpu",
) -> float:
    validate_rashomon_payload(payload)
    states = payload["state"].to(device)
    actions = payload["actions"].to(device)
    actor.eval()
    with torch.no_grad():
        preds = actor(states).argmax(dim=1)
    correct = actions[torch.arange(actions.shape[0], device=actions.device), preds] > 0
    return float(correct.float().mean().item())


def trajectory_preserved(
    actor: torch.nn.Module,
    trajectory_steps: Iterable[TrajectoryStep],
    *,
    env_map: list[str] | tuple[str, ...],
    task_num: float,
    device: str | torch.device = "cpu",
) -> bool:
    for step in trajectory_steps:
        obs = state_index_to_obs(step.state_index, env_map, task_num)
        if greedy_action(actor, obs, device=device) != step.action:
            return False
    return True


def finetune_on_allowed_actions(
    actor: torch.nn.Module,
    payload: RashomonPayload,
    *,
    trajectory_steps: Iterable[TrajectoryStep],
    env_map: list[str] | tuple[str, ...],
    task_num: float,
    lr: float,
    max_epochs: int,
    seed: int,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    """Supervised fine-tune so argmax is always in the allowed action mask."""
    validate_rashomon_payload(payload)
    torch.manual_seed(seed)
    device_t = torch.device(device)
    actor.to(device_t)
    states = payload["state"].to(device_t)
    allowed = payload["actions"].to(device_t).bool()
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    steps_tuple = tuple(trajectory_steps)

    def _metrics() -> tuple[float, bool]:
        actor.eval()
        with torch.no_grad():
            logits = actor(states)
            preds = logits.argmax(dim=1)
            correct = allowed[torch.arange(allowed.shape[0], device=device_t), preds]
        return float(correct.float().mean().item()), trajectory_preserved(
            actor,
            steps_tuple,
            env_map=env_map,
            task_num=task_num,
            device=device_t,
        )

    init_acc, init_preserved = _metrics()
    if init_acc >= 1.0 and init_preserved:
        return {
            "epochs_run": 0,
            "initial_accuracy": init_acc,
            "final_accuracy": init_acc,
            "trajectory_preserved": True,
            "reached_target": True,
        }

    reached = False
    final_acc = init_acc
    preserved = init_preserved
    for epoch in range(1, max_epochs + 1):
        actor.train()
        logits = actor(states)
        masked_logits = logits.masked_fill(~allowed, -1e9)
        log_p_allowed = torch.logsumexp(masked_logits, dim=1) - torch.logsumexp(logits, dim=1)
        loss = -log_p_allowed.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_acc, preserved = _metrics()
        if verbose and (epoch == 1 or epoch % 100 == 0 or (final_acc >= 1.0 and preserved)):
            print(
                f"Safety fine-tune epoch={epoch} loss={float(loss.item()):.6f} "
                f"allowed_acc={final_acc:.3f} trajectory_preserved={preserved}",
            )
        if final_acc >= 1.0 and preserved:
            reached = True
            break

    return {
        "epochs_run": epoch,
        "initial_accuracy": init_acc,
        "final_accuracy": final_acc,
        "trajectory_preserved": preserved,
        "reached_target": reached,
    }
