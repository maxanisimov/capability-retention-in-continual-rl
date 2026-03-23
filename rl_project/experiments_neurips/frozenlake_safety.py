### Imports
import torch
from torch.utils.data import TensorDataset
import numpy as np
import gymnasium
from rl_project.experiments.frozen_lake.train_source_policy import _make_actor, _make_critic, make_frozenlake_env
from rl_project.utils.gymnasium_utils import plot_state_action_pairs
import yaml
import matplotlib.pyplot as plt
from rl_project.utils.gymnasium_utils import plot_state_action_pairs
from rl_project.experiments.frozen_lake.frozenlake_utils import (
    get_all_unsafe_state_action_pairs as fl_get_unsafe_pairs,
)
from rl_project.utils.gymnasium_utils import plot_episode
import inspect
from src.trainer import IntervalTrainer

### Paths
frozenlake_cgf_path = "/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning/rl_project/experiments/frozen_lake/demo_configs.yaml"
results_folder = '/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning/rl_project/experiments/frozen_lake/outputs/standard_4x4_seed42'

### Utils
def create_frozenlake_safety_rashomon_dataset(env, task_flag: float = 0.0):
    """
    Create a TensorDataset containing only safety-critical states and their safe actions.

    - X: one-hot observations (optionally with final task-flag dimension)
    - Y: multi-hot float tensor of shape (N, n_actions) with 1 for safe actions and 0 otherwise
    """
    desc = env.unwrapped.desc
    grid = [
        "".join(ch.decode() if isinstance(ch, (bytes, bytearray)) else str(ch) for ch in row)
        for row in desc
    ]
    nrows, ncols = len(grid), len(grid[0])
    n_states = nrows * ncols
    n_actions = 4  # FrozenLake: Left, Down, Right, Up

    # Infer observation size from env wrapper
    if hasattr(env.observation_space, "shape") and env.observation_space.shape is not None:
        obs_dim_local = int(env.observation_space.shape[0])
    elif hasattr(env.observation_space, "n"):
        obs_dim_local = int(env.observation_space.n)
    else:
        raise ValueError("Cannot infer observation dimension from env.observation_space.")

    if obs_dim_local not in (n_states, n_states + 1):
        raise ValueError(f"Unsupported obs_dim={obs_dim_local}. Expected {n_states} or {n_states + 1}.")

    def state_to_rc(s: int):
        return s // ncols, s % ncols

    def rc_to_state(r: int, c: int):
        return r * ncols + c

    action_deltas = {
        0: (0, -1),  # Left
        1: (1, 0),   # Down
        2: (0, 1),   # Right
        3: (-1, 0),  # Up
    }

    # Identify hole states
    hole_states = set()
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == "H":
                hole_states.add(rc_to_state(r, c))

    obs_list = []
    label_list = []

    for s in range(n_states):
        r, c = state_to_rc(s)
        cell = grid[r][c]

        # Skip terminal/non-traversable states
        if cell in ("H", "G"):
            continue

        safe_actions = []
        for a, (dr, dc) in action_deltas.items():
            nr, nc = r + dr, c + dc
            hits_wall = (nr < 0 or nr >= nrows or nc < 0 or nc >= ncols)

            if hits_wall:
                # In FrozenLake, wall-hit keeps agent in place (safe)
                safe_actions.append(a)
            else:
                ns = rc_to_state(nr, nc)
                if ns not in hole_states:
                    safe_actions.append(a)

        # Keep only safety-critical states (at least one unsafe action exists)
        if len(safe_actions) == n_actions:
            continue

        obs = np.zeros(obs_dim_local, dtype=np.float32)
        obs[s] = 1.0
        if obs_dim_local == n_states + 1:
            obs[-1] = float(task_flag)

        multi_hot = [1.0 if a in safe_actions else 0.0 for a in range(n_actions)]
        obs_list.append(obs)
        label_list.append(multi_hot)

    if len(obs_list) == 0:
        raise RuntimeError("No safety-critical states found; dataset is empty.")

    obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    return TensorDataset(obs_tensor, label_tensor)

def finetune_policy(
    policy: torch.nn.Module,
    dataset,
    env: gymnasium.Env,
    required_accuracy: float = 1.0,
    overlap_mode: str = "safety",
    lr: float = 1e-2,
    max_epochs: int = 3000,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Finetune policy on a combined dataset of safety constraints and trajectory actions.

    Builds a unified allowed-action mask by merging:
    (1) The safety dataset: multi-hot targets indicating which actions are safe per state.
    (2) The policy's own deterministic trajectory: for visited states not in the safety
        dataset, the policy's current argmax action is preserved as the only allowed action.

    Then finetunes with a single objective: maximize log-probability of allowed actions.

    Args:
        policy: The neural network policy to finetune.
        dataset: ``TensorDataset(states, multi_hot_actions)``.  The actions tensor is a
            multi-hot float tensor of shape ``(N, n_actions)`` with 1 for valid actions
            and 0 otherwise.
        env: The environment used to roll out the policy's trajectory.
        required_accuracy: Minimum fraction of states where argmax is an allowed action.
        overlap_mode: How to handle states that appear both in the safety dataset and on
            the trajectory. ``"safety"`` keeps all safe actions from the dataset.
            ``"policy"`` restricts to only the policy's trajectory action.
        lr: Learning rate.
        max_epochs: Maximum training epochs.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        dict with ``policy``, ``final_accuracy``, ``target_accuracy``,
        ``epochs_run``, ``reached_target``, and the ``combined_dataset``.
    """
    if required_accuracy > 1.0:
        required_accuracy = required_accuracy / 100.0
    required_accuracy = float(required_accuracy)

    if overlap_mode not in ("safety", "policy"):
        raise ValueError(f"overlap_mode must be 'safety' or 'policy', got '{overlap_mode}'")

    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise ValueError("Expected a TensorDataset-like object with tensors (X, Y).")

    X, Y = dataset.tensors
    device = next(policy.parameters()).device
    X = X.to(device)
    Y = Y.to(device)

    torch.manual_seed(seed)

    # Infer action dimension
    policy.eval()
    with torch.no_grad():
        n_actions = policy(X[:1]).shape[-1]

    # --- Step 1: Build allowed-action mask from safety dataset ---
    # Y is multi-hot: shape (N, n_actions) with 1s for valid actions
    state_to_allowed: dict[tuple, set] = {}
    for i in range(X.shape[0]):
        key = tuple(X[i].detach().cpu().tolist())
        if key not in state_to_allowed:
            state_to_allowed[key] = set()
        valid_actions = torch.where(Y[i] > 0)[0].tolist()
        state_to_allowed[key].update(valid_actions)

    # --- Step 2: Roll out policy trajectory and merge ---
    dataset_keys = set(state_to_allowed.keys())
    obs, _ = env.reset(seed=seed)
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(policy(obs_t).argmax(dim=1).item())
            key = tuple(obs_t.squeeze(0).cpu().tolist())
            if key not in dataset_keys:
                # State only on trajectory: preserve the policy's action
                state_to_allowed[key] = {action}
            elif overlap_mode == "policy":
                # State in both: restrict to only the policy's action
                state_to_allowed[key] = {action}
            # overlap_mode == "safety": keep the dataset's allowed actions unchanged
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    # --- Step 3: Build combined tensors ---
    keys = list(state_to_allowed.keys())
    combined_states = torch.tensor(keys, dtype=X.dtype, device=device)
    allowed_mask = torch.zeros(len(keys), n_actions, dtype=torch.bool, device=device)
    for i, key in enumerate(keys):
        for a in state_to_allowed[key]:
            allowed_mask[i, a] = True

    n_total = combined_states.shape[0]
    if verbose:
        n_safety = X.shape[0]
        n_combined = n_total
        print(f"\n--- Finetuning policy ---")
        print(f"  Safety dataset states: {n_safety}")
        print(f"  Combined dataset states (safety + trajectory): {n_combined}")

    # --- Early exit check ---
    with torch.no_grad():
        logits0 = policy(combined_states)
        preds0 = logits0.argmax(dim=1)
        init_acc = float(
            allowed_mask[torch.arange(n_total, device=device), preds0].float().mean().item()
        )

    if init_acc >= required_accuracy:
        if verbose:
            print(f"  Already satisfies target | acc={init_acc:.3f} (target={required_accuracy:.3f})")
        return {
            "policy": policy,
            "final_accuracy": init_acc,
            "target_accuracy": required_accuracy,
            "epochs_run": 0,
            "reached_target": True,
            "combined_dataset": TensorDataset(combined_states, allowed_mask.float()),
        }

    # --- Finetune ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    reached = False
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        logits = policy(combined_states)
        safe_logits = logits.masked_fill(~allowed_mask, -1e9)
        log_p_allowed = torch.logsumexp(safe_logits, dim=1) - torch.logsumexp(logits, dim=1)
        loss = -log_p_allowed.mean()

        preds = logits.argmax(dim=1)
        acc = allowed_mask[torch.arange(n_total, device=device), preds].float().mean()
        acc_v = float(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 100 == 0 or acc_v >= required_accuracy):
            print(f"  Epoch {epoch:4d} | loss={loss.item():.6f} | acc={acc_v:.3f}")

        if acc_v >= required_accuracy:
            reached = True
            break

    # --- Final evaluation ---
    policy.eval()
    with torch.no_grad():
        logits = policy(combined_states)
        preds = logits.argmax(dim=1)
        final_acc = float(
            allowed_mask[torch.arange(n_total, device=device), preds].float().mean().item()
        )

    if verbose:
        print(f"  Final | acc={final_acc:.3f} (target={required_accuracy:.3f}) | reached={reached}")
        print("--- Finetuning complete ---\n")

    if not reached:
        raise RuntimeError(
            "Could not satisfy constraints within max_epochs. "
            "Try larger max_epochs or lower lr."
        )

    return {
        "policy": policy,
        "final_accuracy": final_acc,
        "target_accuracy": required_accuracy,
        "epochs_run": epoch,
        "reached_target": reached,
        "combined_dataset": TensorDataset(combined_states, allowed_mask.float()),
    }

def compute_margin_surrogate_threshold(
    n_allowed_actions_per_state: list[int],
    T: float = 10.0,
) -> dict:
    """
    Compute the per-state and aggregate LSE margin thresholds needed to
    certify hard specification.

    A state is certified safe when its LSE margin surrogate exceeds
    τ·log(K_safe), where τ = 1/T and K_safe is the number of allowed (safe) actions
    in that state. This follows from the LSE upper-bound property:

        lse_τ(z(s)) ≤ max(z(s)) + τ·log(K_safe)

    applied to the safe-action side of the margin.

    Args:
        n_allowed_actions_per_state: Number of allowed actions for each critical state.
        T: Temperature parameter (same as SOFT_ACC_TEMP in interval_utils).

    Returns:
        Dictionary with per-state thresholds and aggregate statistics.
    """
    tau = 1.0 / T
    per_state = [tau * np.log(k) if k > 1 else 0.0 for k in n_allowed_actions_per_state]
    return {
        "tau": tau,
        "per_state_thresholds": per_state,
        "mean_threshold": float(np.mean(per_state)),
        "max_threshold": float(np.max(per_state)),
    }

### Calculate the surrogate LSE margins for the source policy in the safety-critical states, and compare to the thresholds
def lse_margin_surrogate(logits: torch.Tensor, safe_action_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Compute the LSE margin surrogate for a given set of action logits and
    a mask indicating which actions are safe.

    The surrogate is defined as:

        lse_τ(S_safe) - max(S_unsafe)
    where S_safe are the logits of the safe actions, S_unsafe are the logits of the unsafe actions, and lse_τ is the temperature-scaled log-sum-exp function:
        lse_τ(S) = τ * log(sum(exp(S / τ)))
    Args:
        logits: Tensor of shape (n_actions,) containing the action logits.
        safe_action_mask: Boolean tensor of shape (n_actions,) where True indicates a safe action.
        tau: Temperature parameter.
    Returns:
        The LSE margin surrogate value.
    """    
    safe_logits = logits[safe_action_mask]
    unsafe_logits = logits[~safe_action_mask]

    if len(safe_logits) == 0:
        raise ValueError("No safe actions provided.")
    if len(unsafe_logits) == 0:
        # If there are no unsafe actions, the margin is effectively infinite
        return torch.tensor(float('inf'))

    lse_safe = tau * torch.logsumexp(safe_logits / tau, dim=0)
    max_unsafe = torch.max(unsafe_logits)

    margin_surrogate = lse_safe - max_unsafe
    return margin_surrogate


# ####### Load ############################
# ### Load the policy
# policy_path = f'{results_folder}/source_policy.pt'
# source_actor_state_dict = torch.load(policy_path, map_location='cpu')

# ### Load the critic
# critic_path = f'{results_folder}/source_critic.pt'
# source_critic_state_dict = torch.load(critic_path, map_location='cpu')

# ### Load training data of the source policy, which is needed for EWC PPO training. 
# source_training_data_path = f'{results_folder}/source_training_data.pt'
# source_training_data = torch.load(source_training_data_path, map_location='cpu', weights_only=False)

# ### Load configuration for the environment
# with open(frozenlake_cgf_path) as f:
#     all_cfgs = yaml.safe_load(f)

# cfg = all_cfgs['standard_4x4']
# env1_map = cfg['env1_map']
# env2_map = cfg['env2_map']
# env = make_frozenlake_env(env1_map, task_num=0, is_slippery=False)
# obs_dim = env.observation_space.shape[0]  # num_states + 1
# n_states = obs_dim - 1  # last dim is the task flag

# ### Source actor
# source_actor = _make_actor(obs_dim, hidden=64)
# source_actor.load_state_dict(source_actor_state_dict)

# ### Source critic
# source_critic = _make_critic(obs_dim, hidden=64)
# source_critic.load_state_dict(source_critic_state_dict)

# ####### Experiment ############################

# ### OPTIONAL: Plot the source policy actions in each state

# # # Generate source policy action for every state
# # source_state_action_pairs = []
# # for state_idx in range(n_states):
# #     obs = np.zeros(obs_dim, dtype=np.float32)
# #     obs[state_idx] = 1.0  # one-hot encoding, task flag = 0
# #     with torch.no_grad():
# #         logits = source_actor(torch.tensor(obs).unsqueeze(0))
# #         action = logits.argmax(dim=1).item()
# #     source_state_action_pairs.append((state_idx, action))

# # # Plot on the FrozenLake grid
# # env_plot = gymnasium.make("FrozenLake-v1", desc=env1_map, render_mode="rgb_array")
# # fig = plot_state_action_pairs(
# #     env=env_plot,
# #     state_action_pairs=source_state_action_pairs,
# #     arrow_color="black",
# #     title="Source policy actions (Task 1)",
# # )
# # env_plot.close()

# ### Create the Rashomon dataset for safety-critical states and their safe actions
# rashomon_dataset = create_frozenlake_safety_rashomon_dataset(env, task_flag=0.0)

# # # OPTIONAL: Visualize the safety-critical state-action pairs
# # safety_state_action_pairs = []
# # for obs, label in safety_rashomon_dataset:
# #     state_idx = obs[:-1].argmax().item()  # get state index from one-hot (ignore task flag)
# #     safe_actions = [a.item() for a in label if a.item() != -1]
# #     for a in safe_actions:
# #         safety_state_action_pairs.append((state_idx, a))
# # env_plot = gymnasium.make("FrozenLake-v1", desc=env1_map, render_mode="rgb_array")
# # fig = plot_state_action_pairs(
# #     env=env_plot,
# #     state_action_pairs=safety_state_action_pairs,
# #     arrow_color="blue",
# #     title="Safety-critical state-action pairs (Task 1)",
# # )
# # env_plot.close()

# ### Finetune the source policy to the Rashomon dataset if needed
# finetune_policy_output = finetune_policy(
#     policy=source_actor,
#     dataset=rashomon_dataset,
#     env=env,
#     required_accuracy=1.0,
#     lr=1e-2,
#     max_epochs=3000,
#     seed=42,
#     verbose=True,
# )
# source_actor_finetuned = finetune_policy_output['policy']

# # # OPTIONAL: Plot actions of source_actor_finetuned on the FrozenLake grid
# # source_finetuned_state_action_pairs = []
# # for state_idx in range(obs_dim - 1):  # exclude task-flag dimension
# #     obs = np.zeros(obs_dim, dtype=np.float32)
# #     obs[state_idx] = 1.0
# #     with torch.no_grad():
# #         logits = source_actor_finetuned(torch.tensor(obs).unsqueeze(0))
# #         action = int(logits.argmax(dim=1).item())
# #     source_finetuned_state_action_pairs.append((state_idx, action))

# # env_plot_finetuned = gymnasium.make("FrozenLake-v1", desc=env1_map, render_mode="rgb_array")
# # fig_source_finetuned = plot_state_action_pairs(
# #     env=env_plot_finetuned,
# #     state_action_pairs=source_finetuned_state_action_pairs,
# #     arrow_color="blue",
# #     title="Actions of Source Finetuned Policy (Task 1)",
# # )
# # plt.show()
# # env_plot_finetuned.close()


# ### Calibrate the softmax temperature so that the source policy satisfies the surrogate specification constraint
# # Compute for our safety-critical states
# n_allowed_per_state = [
#     int(rashomon_dataset.tensors[1][i, :].sum().item()) for i in range(rashomon_dataset.tensors[1].shape[0])
# ]
# thresholds = compute_margin_surrogate_threshold(n_allowed_per_state, T=10.0)
# # Identify safety-critical states from the Rashomon labels:
# # a state is critical if it has at least one disallowed action.
# n_actions = env.action_space.n
# nrows, ncols = len(env1_map), len(env1_map[0])

# def state_to_rc(s: int, ncols: int):
#     return s // ncols, s % ncols

# state_to_safe_actions = {}
# for obs_i, multi_hot_i in rashomon_dataset:
#     s = int(obs_i[:n_states].argmax().item())
#     allowed = torch.where(multi_hot_i > 0)[0].tolist()
#     state_to_safe_actions[s] = sorted(allowed)

# safety_critical_states = sorted(
#     [s for s, allowed in state_to_safe_actions.items() if len(allowed) < n_actions]
# )
# safe_actions_per_state = [state_to_safe_actions[s] for s in safety_critical_states]
# n_safe_per_state = [len(a) for a in safe_actions_per_state]
# max_num_allowed_actions = max(n_safe_per_state)
# print(f"\nMax number of allowed actions in any critical state: {max_num_allowed_actions}")

# # print("Safety-critical states:", safety_critical_states)
# # for s, allowed in zip(safety_critical_states, safe_actions_per_state):
# #     r, c = state_to_rc(s, ncols)
# #     disallowed = sorted(set(range(n_actions)) - set(allowed))
# #     print(f"  State {s} (r={r}, c={c}) | allowed={allowed} | disallowed={disallowed}")

# # print(f"τ = 1/T = {thresholds['tau']}")
# # for s, n_allowed, thr in zip(safety_critical_states, n_allowed_per_state, thresholds["per_state_thresholds"]):
# #     r, c = state_to_rc(s, ncols)
# #     print(f"  State {s} (r={r}, c={c}): {n_allowed} allowed actions → threshold = τ·log({n_allowed}) = {thr:.6f}")
# # print(f"\nMean threshold:  {thresholds['mean_threshold']:.6f}")
# # print(f"Pessimistic threshold (max over states): {thresholds['max_threshold']:.6f}")

# max_allowed_actions_per_state = max(n_safe_per_state)
# print(f"Using max_allowed_actions_per_state = {max_allowed_actions_per_state} for global threshold calculation.")

# # Find the largest certified surrogate temperature (tau) for the source policy
# # Certification condition per safety-critical state:
# #   lse_margin_surrogate(logits, safe_mask, tau) > tau * log(K_safe)

# # 1) Precompute logits/masks for safety-critical states
# critical_data = []
# for s, safe_acts in zip(safety_critical_states, safe_actions_per_state):
#     obs = np.zeros(obs_dim, dtype=np.float32)
#     obs[s] = 1.0
#     obs_t = torch.tensor(obs).unsqueeze(0)

#     with torch.no_grad():
#         logits_s = source_actor(obs_t).squeeze(0)

#     safe_mask_s = torch.zeros(n_actions, dtype=torch.bool)
#     safe_mask_s[safe_acts] = True

#     critical_data.append((s, safe_acts, logits_s, safe_mask_s))

# def all_states_certified(tau: float, use_global_threshold: bool = False, strict: bool = True) -> bool:
#     """
#     If use_global_threshold=True, uses tau*log(max_num_safe_actions) for every state.
#     Otherwise, uses per-state tau*log(len(safe_acts)).
#     """
#     for s, safe_acts, logits_s, safe_mask_s in critical_data:
#         margin = lse_margin_surrogate(logits_s, safe_mask_s, tau=tau).item()
#         k = max_allowed_actions_per_state if use_global_threshold else len(safe_acts)
#         thr = tau * np.log(k)
#         if strict:
#             if not (margin > thr):
#                 return False
#         else:
#             if not (margin >= thr):
#                 return False
#     return True

# # 2) Bracket the maximum feasible tau
# use_global_threshold = True  # set True for pessimistic global threshold
# tau_min = 1e-8
# tau_cap = 10.0

# if not all_states_certified(tau_min, use_global_threshold=use_global_threshold, strict=True):
#     raise RuntimeError("Even very small tau is not certified. Check policy/safety setup.")

# tau_lo = tau_min
# tau_hi = 0.05  # initial probe

# while tau_hi < tau_cap and all_states_certified(tau_hi, use_global_threshold=use_global_threshold, strict=True):
#     tau_lo = tau_hi
#     tau_hi *= 2.0

# if tau_hi >= tau_cap and all_states_certified(tau_cap, use_global_threshold=use_global_threshold, strict=True):
#     best_tau = tau_cap
# else:
#     # 3) Binary search for largest certified tau in [tau_lo, tau_hi]
#     for _ in range(60):
#         mid = 0.5 * (tau_lo + tau_hi)
#         if all_states_certified(mid, use_global_threshold=use_global_threshold, strict=True):
#             tau_lo = mid
#         else:
#             tau_hi = mid
#     best_tau = tau_lo

# best_T = 1.0 / best_tau

# print(f"Largest certified tau: {best_tau:.10f}")
# print(f"Equivalent T = 1/tau: {best_T:.6f}")
# print(f"Threshold mode: {'global max K_safe' if use_global_threshold else 'per-state K_safe'}")

# # 4) Optional: show per-state margins at best_tau
# print("\nPer-state check at best tau:")
# for s, safe_acts, logits_s, safe_mask_s in critical_data:
#     margin = lse_margin_surrogate(logits_s, safe_mask_s, tau=best_tau).item()
#     k = max_allowed_actions_per_state if use_global_threshold else len(safe_acts)
#     thr = best_tau * np.log(k)
#     r, c = state_to_rc(s, ncols)
#     status = "OK" if margin > thr else "FAIL"
#     print(
#         f"  State {s} (r={r}, c={c}), K_safe={len(safe_acts)}: "
#         f"margin={margin:.6f}, threshold={thr:.6f} -> {status}"
#     )

# print(f"\nSource policy is certified safe with T ≤ {best_T:.6f} under the {'global' if use_global_threshold else 'per-state'} threshold.")

# # Calibrate temperature for source_actor_finetuned so the soft safety constraint holds
# # Constraint per safety-critical state:
# #   lse_margin_surrogate(logits, safe_mask, tau) > tau * log(K)
# # with K = max_num_safe_actions (global pessimistic threshold) by default.

# # 1) Precompute logits/masks for safety-critical states using the FINETUNED policy
# critical_data_finetuned = []
# for s, safe_acts in zip(safety_critical_states, safe_actions_per_state):
#     obs = np.zeros(obs_dim, dtype=np.float32)
#     obs[s] = 1.0
#     obs_t = torch.tensor(obs).unsqueeze(0)

#     with torch.no_grad():
#         logits_s = source_actor_finetuned(obs_t).squeeze(0)

#     safe_mask_s = torch.zeros(n_actions, dtype=torch.bool)
#     safe_mask_s[safe_acts] = True
#     critical_data_finetuned.append((s, safe_acts, logits_s, safe_mask_s))

# def all_states_certified_finetuned(tau: float, use_global_threshold: bool = True, strict: bool = True) -> bool:
#     for s, safe_acts, logits_s, safe_mask_s in critical_data_finetuned:
#         margin = lse_margin_surrogate(logits_s, safe_mask_s, tau=tau).item()
#         k = max_allowed_actions_per_state if use_global_threshold else len(safe_acts)
#         thr = tau * np.log(k)
#         if strict:
#             if not (margin > thr):
#                 return False
#         else:
#             if not (margin >= thr):
#                 return False
#     return True

# # 2) Find largest certified tau by bracketing + binary search (ONLY IF THE SURROGATE SPECIFICATION IS NOT SATISFIED at proposed tau for source policy)
# tau = 1/10
# # Check source policy surrogate constraint at tau = 1/10
# use_global_threshold = True
# source_certified_at_tau = all_states_certified(
#     tau,
#     use_global_threshold=use_global_threshold,
#     strict=True,
# )

# print(
#     f"Source policy surrogate constraint at tau={tau:.6f} "
#     f"({'global' if use_global_threshold else 'per-state'} threshold): "
#     f"{'SATISFIED' if source_certified_at_tau else 'NOT SATISFIED'}"
# )

# best_T_finetuned = 1/ tau
# best_tau_finetuned = tau
# if not source_certified_at_tau:

#     use_global_threshold = True
#     tau_min = 1e-8
#     tau_cap = tau

#     if not all_states_certified_finetuned(tau_min, use_global_threshold=use_global_threshold, strict=True):
#         raise RuntimeError("Even very small tau is not certified for source_actor_finetuned.")

#     tau_lo = tau_min
#     tau_hi = 0.05
#     while tau_hi < tau_cap and all_states_certified_finetuned(tau_hi, use_global_threshold=use_global_threshold, strict=True):
#         tau_lo = tau_hi
#         tau_hi *= 2.0

#     if tau_hi >= tau_cap and all_states_certified_finetuned(tau_cap, use_global_threshold=use_global_threshold, strict=True):
#         best_tau_finetuned = tau_cap
#     else:
#         for _ in range(60):
#             mid = 0.5 * (tau_lo + tau_hi)
#             if all_states_certified_finetuned(mid, use_global_threshold=use_global_threshold, strict=True):
#                 tau_lo = mid
#             else:
#                 tau_hi = mid
#         best_tau_finetuned = tau_lo

# best_T_finetuned = 1.0 / best_tau_finetuned
# margin_surrogate_threshold_finetuned = best_tau_finetuned * np.log(max_allowed_actions_per_state)

# print(f"Largest certified tau (finetuned): {best_tau_finetuned:.12f}")
# print(f"Equivalent temperature T=1/tau: {best_T_finetuned:.6f}")
# print(f"Global threshold tau*log(max_allowed_actions_per_state={max_allowed_actions_per_state}): {margin_surrogate_threshold_finetuned:.12f}")

# print("\nPer-state check at best_tau_finetuned:")
# for s, safe_acts, logits_s, safe_mask_s in critical_data_finetuned:
#     margin = lse_margin_surrogate(logits_s, safe_mask_s, tau=best_tau_finetuned).item()
#     k = max_allowed_actions_per_state if use_global_threshold else len(safe_acts)
#     thr = best_tau_finetuned * np.log(k)
#     r, c = state_to_rc(s, ncols)
#     status = "OK" if margin > thr else "FAIL"
#     print(
#         f"  State {s} (r={r}, c={c}), K_safe={len(safe_acts)}: "
#         f"margin={margin:.6f}, threshold={thr:.6f} -> {status}"
#     )

# # Optional: overwrite the shared variables used in later cells
# best_tau = best_tau_finetuned
# best_T = best_T_finetuned

# ##### RASHOMON SET COMPUTATION ############
# margin_surrogate_threshold = best_tau * np.log(max_allowed_actions_per_state)  # since the max number of safe actions is 2 in our case
# min_soft_specification_value = margin_surrogate_threshold

# trainer = IntervalTrainer(
#     model=source_actor_finetuned,
#     min_acc_limit=min_soft_specification_value, # worst-case margin threshold across states
#     seed=42,
#     n_iters=5_000, # NOTE: running for too long can breach the constraint and coming back can be hard # TODO: you can use checkpoints
#     min_acc_increment=0,
#     soft_acc_temperature=best_T,
#     aggregation="min",  # constrain worst-case per-state margin (sound for safety)
#     checkpoint=100, # save a checkpointed bound every checkpoint iterations,
# )

# print("Computing Rashomon set...")
# trainer.compute_rashomon_set(
#     dataset=rashomon_dataset, 
#     multi_label=True
# )
# print("Done.")

# # # OPTIONAL: Plot the hard certification progress over iterations
# # iters_num = [20 * i for i in range(len(trainer.certificates))]
# # plt.figure(figsize=(12, 5))
# # plt.plot(iters_num, trainer.certificates)
# # plt.xlabel("Iteration")
# # plt.ylabel("Certified worst-case safety rate")
# # plt.title("Certification progress over iterations")
# # plt.grid()

# largest_certified_rashomonset_idx = -1
# if trainer.certificates[-1] < 1.0:
#     largest_certified_rashomonset_idx = [i for i, certificate in enumerate(trainer.certificates) if trainer.certificates[i] == 1.0 and trainer.certificates[i+1] < 1.0][0]
# final_certificate = trainer.certificates[largest_certified_rashomonset_idx]
# bounded_model = trainer.bounds[largest_certified_rashomonset_idx]
# print(f'Hard specification = 1 certified with {largest_certified_rashomonset_idx} checkpoints.')

# ### Create X_crit
# is_slippery = False
# def make_env(task: int, render_mode: str | None = None) -> gymnasium.Env:
#     return make_frozenlake_env(
#         env_map=env1_map if task == 0 else env2_map,
#         task_num=task,
#         is_slippery=is_slippery,
#         render_mode=render_mode
#     )

# def get_unsafe_pairs(task: int):
#     env_map = env1_map if task == 0 else env2_map
#     pairs = fl_get_unsafe_pairs(env_map=env_map, task_num=task, state_repr="observation")
#     return [(np.asarray(s, dtype=np.float32), int(a)) for s, a in pairs]
# import pandas as pd

# def state_key(obs: np.ndarray) -> tuple[float, ...]:
#     return tuple(np.asarray(obs, dtype=np.float32).round(6).tolist())

# def unsafe_pairs_to_critical_map(
#     unsafe_pairs: list[tuple[np.ndarray, int]],
# ):
#     crit = {}
#     for obs, action in unsafe_pairs:
#         key = state_key(obs)
#         if key not in crit:
#             crit[key] = {"obs": np.asarray(obs, dtype=np.float32), "unsafe_actions": set()}
#         crit[key]["unsafe_actions"].add(int(action))
#     return crit

# # Build critical maps for both tasks
# critical_maps = {}
# for t in (0, 1):
#     pairs = get_unsafe_pairs(t)
#     critical_maps[t] = unsafe_pairs_to_critical_map(pairs)
#     print(f"Task {t+1}: {len(critical_maps[t])} critical states, {len(pairs)} unsafe (state, action) pairs")
# action_labels = [f"A{i}" for i in range(n_actions)]

# # Collect all critical states from task 1
# critical_states = []
# critical_keys = []
# for key, entry in critical_maps[0].items():
#     critical_states.append(entry["obs"])
#     critical_keys.append(key)

# X_crit = torch.as_tensor(np.stack(critical_states), dtype=torch.float32)
# n_crit = X_crit.shape[0]


# ### Plot the Rashomon set actions
# # Extract bounds and certificates
# bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
# bounds_u = [p.detach().cpu() for p in bounded_model.param_u]

# # Propagate X_crit through IBP bounds to get min/max logits across Rashomon set
# bounded_model_cpu = bounded_model.to("cpu")
# with torch.no_grad():
#     logits_l, logits_u = bounded_model_cpu.bound_forward(X_crit, X_crit)
#     prob_l = torch.softmax(logits_l, dim=1).numpy()
#     prob_u = torch.softmax(logits_u, dim=1).numpy()

# all_obs_task1 = np.array([
#     np.eye(n_states, dtype=np.float32)[s] for s in range(n_states)
# ])
# # Append task flag (0.0) to match obs_dim
# all_obs_task1 = np.concatenate([all_obs_task1, np.zeros((n_states, 1), dtype=np.float32)], axis=1)
# all_obs_task1_t = torch.tensor(all_obs_task1, dtype=torch.float32)
# logit_bounds = bounded_model_cpu.bound_forward(all_obs_task1_t, all_obs_task1_t)
# # logit_bounds[0] # lower bounds
# # logit_bounds[1] # upper bounds

# rashomon_actions_per_state_idx_dct = {}
# for cur_state_idx in range(n_states-1): # NOTE: skip the task label!

#     with torch.no_grad():
#         cur_logits_lower = logit_bounds[0][cur_state_idx].numpy()
#         cur_logits_upper = logit_bounds[1][cur_state_idx].numpy()

#     # if the action's upper bound is below any lower bound, this action is NOT feasible in the Rashmon set
#     feasible_actions = []
#     for action_idx in range(n_actions):
#         other_actions = [a for a in range(n_actions) if a != action_idx]
#         # print(action_idx)
#         # print(cur_logits_upper[action_idx] < cur_logits_lower[other_actions])
#         if any(cur_logits_upper[action_idx] < cur_logits_lower[other_actions]):
#             continue
#         else:
#             feasible_actions.append(action_idx)
#     # print(f"Feasible actions for state {cur_state_idx}: {feasible_actions}")
#     rashomon_actions_per_state_idx_dct[cur_state_idx] = feasible_actions

# rashomon_state_action_pairs = []
# for state_idx in rashomon_actions_per_state_idx_dct:
#     for action in rashomon_actions_per_state_idx_dct[state_idx]:
#         rashomon_state_action_pairs.append([state_idx, action])

# env_plot = gymnasium.make("FrozenLake-v1", render_mode="rgb_array")

# rashomon_actions_fig = plot_state_action_pairs(
#     env=env_plot,
#     state_action_pairs=rashomon_state_action_pairs,
#     arrow_color="royalblue",
#     title=f"Rashomon set actions",
# )