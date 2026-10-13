# RL-SGF update modes

New RL-SGF training defaults to **RL-SGF-PPO**. Select the original algorithm with
`--rl-sgf-update-mode reinforce`; `--rl-sgf-update-mode ppo` explicitly requests
the new default. The stage, initialisation-ablation sweep, projected-baseline
sweep and timing launcher expose this option.

## PPO proposals and SGF constraints

Each iteration collects a fixed-length PPO rollout (`--n-steps`) and a separate
batch of complete episodes (`--rl-sgf-episodes-per-iter`) under the same
pre-update policy. Constraint episodes use an independent environment with the
same configuration and seed offset 60,000; they do not reset the PPO environment.
Their discounted cost value and REINFORCE cost gradient use the original SGF
estimator.

The reward proposal uses GAE, the clipped PPO objective, entropy regularisation,
reward-critic fitting and all requested epochs/minibatches. It does not use a
Lagrange multiplier. After the epochs, the full actor proposal is filtered once
by the existing SGF constraint. Critics and Adam moments are retained. An
infeasible constraint estimate retains the original cost-descent fallback.

This is a hybrid variant; the original RL-SGF paper's guarantees are not claimed
for it. `--rl-sgf-baseline none` disables Monte Carlo critic baselines, but the
PPO proposal still fits a reward critic.

The timestep budget counts **both** PPO and constraint-sampling interactions.
The final complete iteration can exceed the nominal budget. Consequently equal
total budgets need not yield equal numbers of PPO updates. Results record
`ppo_timesteps`, `constraint_timesteps`, `reinforce_timesteps`,
`ppo_minibatch_updates`, `update_count` and the selected mode. Their sampling
counts sum to the actual training steps. Timing records also include SGF case
counts and the last proposal/accepted displacement norms. `training_loop_s`
includes both sampling phases, PPO optimisation, SGF estimation/filtering, and
periodic evaluation.

## Launch examples

From the repository root, this starts a fresh RL-SGF-PPO timing sweep across all
six MASA environments and seeds 0–9:

```bash
MPLCONFIGDIR=/tmp/rl-sgf-mpl .venv/bin/python \
  projects/safe_policy_optimisation/scripts/experiments/run_timed_main_comparison.py \
  --methods rl_sgf --rl-sgf-update-mode ppo --max-parallel 16 \
  --output-dir projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/rl_sgf_ppo_timing_NEW_RUN \
  --launch
```

Use `--rl-sgf-update-mode reinforce` and a different output directory to time the
original algorithm. Timing sweeps reuse the saved safe actor and exclude its
original fitting cost. Add other method names to `--methods` for a comparison.
Per-job JSON and the completed sweep's `training_times.csv` record the mode and
method label separately from the stable algorithm key `rl_sgf`.

For ordinary training, use `stages/train_rl_sgf.py` with the same mode flag. Its
default episode batch is 100; the MASA sweep/timing configurations retain 20.
In the initialisation-ablation and projected-baseline sweeps, PPO-based runs use
the `rl_sgf_ppo` method directory; native runs retain `rl_sgf`. Existing results
must match the requested mode before they can be reused.

## Historical results

New timing manifests explicitly pin their requested mode, even when their
reference configurations predate the flag. Existing manifests with no mode
retain **REINFORCE**, including remaining jobs in the active baseline timing
sweep. Existing checkpoints with no mode also load as REINFORCE; new checkpoints
save the selected mode, hyperparameters and sampling counters.

Historical plot inputs remain attached to their original cohorts. Changing the
training default does not relabel those results or replace their saved timings.
