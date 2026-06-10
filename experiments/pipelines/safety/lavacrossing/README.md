# LavaCrossing Shield Safety Pipeline

`CustomLavaCrossing-v0` continual-learning pipeline with deterministic and stochastic dynamics selected by pipeline key or `--slip-prob` override.

- Source training with shield-derived multi-hot safety targets.
- Downstream baselines: unconstrained PPO, EWC, Rashomon, SafeLineSearch PPO, and Lagrangian PPO.
- Observations are `[normalized_row, normalized_col, task_id]`.
- Actions are `0=left`, `1=right`, `2=down`, `3=up`, `4=stay`.
- Stochastic dynamics use `slip_prob`: the intended action is taken with probability `1 - slip_prob`; the remaining probability is split across the other actions.
- Lava cells are unsafe terminals, goal cells are successful terminals, and wall cells are non-traversable.
- Probabilistic shield source datasets use minimum-risk actions from value iteration, independent of `shield_risk_threshold`.

Initial pipeline keys:

- `corridor_7x7_deterministic`
- `corridor_7x7_slip_0p1`
- `route_switch_7x7_deterministic`
- `route_switch_7x7_slip_0p1`

Run one source job:

```bash
python -m experiments.pipelines.safety.lavacrossing.run_experiment \
  --mode source \
  --pipeline corridor_7x7_deterministic \
  --seed 0
```

Run a stochastic full multi-seed pipeline:

```bash
python -m experiments.pipelines.safety.lavacrossing.cli.launch_full_pipeline_multi_seed \
  --pipeline corridor_7x7_slip_0p1 \
  --seeds 0 1 2 \
  --cores 0 1 2
```

Override slip probability for an experiment:

```bash
python -m experiments.pipelines.safety.lavacrossing.run_experiment \
  --mode source \
  --pipeline corridor_7x7_deterministic \
  --seed 0 \
  --slip-prob 0.2
```

Run the verified-margin baselines:

```bash
python -m experiments.pipelines.safety.lavacrossing.run_safe_line_search_ppo \
  --pipeline corridor_7x7_deterministic \
  --seed 0

python -m experiments.pipelines.safety.lavacrossing.run_lagrangian_ppo \
  --pipeline corridor_7x7_deterministic \
  --seed 0
```

Aggregate results:

```bash
python -m experiments.pipelines.safety.lavacrossing.aggregate_metrics_lavacrossing_shield_safety \
  --pipeline corridor_7x7_deterministic
```

Plot initial frames or a synthesized shield without training:

```bash
python -m experiments.pipelines.safety.lavacrossing.plot_initial_frames \
  --pipeline route_switch_7x7_slip_0p1

python -m experiments.pipelines.safety.lavacrossing.plot_synthesised_shield \
  --pipeline corridor_7x7_slip_0p1 \
  --shield-risk-threshold 0.05
```

Artifacts default to `experiments/pipelines/safety/lavacrossing/artifacts/runs/<pipeline_key>`.
Every run summary records the resolved environment setup, including `env_id`, `dynamics`, `slip_prob`, maps, action set, action count, episode budget, and shield settings.
