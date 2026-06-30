# Safe Policy Optimisation

Experiments for safe policy optimisation.

## Development setup

Install the repository as an editable package once. This puts both the
`core/` libraries and the `projects/` tree on the import path, so the stage
scripts and `run_experiment.py` work from any working directory without the
`sys.path` shims they used to carry:

```bash
pip install -e .          # from the repository root
```

This requires a recent build toolchain (`setuptools>=64`, `pip>=21.3`) for the
PEP 660 editable install; upgrade with `pip install -U pip setuptools wheel` if
`import projects.safe_policy_optimisation` is not found after installing.

Run the test suite with the standard library test runner:

```bash
python -m unittest discover -s projects/safe_policy_optimisation/tests -p "test_*.py"
```

## Project structure

The project is organised around reusable helpers, declarative experiment
settings, runnable pipelines, and generated outputs:

```text
projects/safe_policy_optimisation/
  run_experiment.py        # preferred launcher for full experiment pipelines
  settings/deterministic/{tasks,pipelines}.yaml
  settings/paper_2503_07671/{tasks,pipelines}.yaml
  stages/                  # pipeline/stage implementations (thin CLI wrappers)
  utils/                   # shared helper modules:
    safe_rl.py             #   safe-RL baseline factories, evaluation, checkpoints
    io.py                  #   result IO: JSON / episode CSV writers + row builders
    metrics.py             #   summarise_evaluation(): success / reward / safety
    shield.py              #   load_shield_mask() for shield_q.pt artifacts
    envs.py                #   parse_env_kwargs() / env_kwargs_from_args()
    cli.py                 #   shared argparse blocks (PPO hyperparameters)
    seeding.py             #   set_global_seeds() + seed-offset constants
    log.py                 #   logging setup (per-stage log capture compatible)
    safe_crl_bridge.py     #   single adapter for the safe_crl cross-project import
    config.py              #   YAML pipeline/task settings loader
    config_schema.py       #   typed dataclass schema + validation for settings
    learning_curves.py     #   TensorBoard / CSV learning-curve logging
    cpu_allocation.py      #   CPU affinity / worker-pool sizing
  outputs/                 # per-run results (gitignored)
  artifacts/               # reusable cross-run inputs: shields, Rashomon sets, rollouts (gitignored)
  tests/                   # project tests
```

`artifacts/` holds reusable, cross-run inputs (synthesised shields, Rashomon
sets, rollout GIFs); `outputs/` holds the per-run results of a launcher run.
Both are gitignored.

For new full-pipeline runs, prefer the launcher:

```bash
python projects/safe_policy_optimisation/run_experiment.py \
  --pipeline deterministic_minipacman
```

List registered pipelines with:

```bash
python projects/safe_policy_optimisation/run_experiment.py --list-pipelines
```

Paper-scale settings for arXiv:2503.07671 are registered with the
`paper_2503_07671_*` prefix. For example:

```bash
python projects/safe_policy_optimisation/run_experiment.py \
  --pipeline paper_2503_07671_colour_bomb \
  --force-shield-synthesis
```

These settings keep the local gridworld implementation's five actions
(`left`, `right`, `down`, `up`, and `stay`), so gridworld action spaces differ
from the four-action table in the paper while preserving this repo's dynamics.

Task and pipeline settings live in two grouped files:

```text
projects/safe_policy_optimisation/settings/deterministic/{tasks,pipelines}.yaml
projects/safe_policy_optimisation/settings/paper_2503_07671/{tasks,pipelines}.yaml
```

Each `pipelines.yaml` keeps the shared body once under a `_defaults:` block and
pulls it into each pipeline with native YAML anchors / merge keys
(`runtime: *runtime`, `training: {<<: *training, total_timesteps: 200000}`), so a
new pipeline only specifies what differs. Settings are validated against the
typed schema in `utils/config_schema.py` before a run starts: unknown sections or
fields, missing required keys, and typos fail fast with a clear message instead
of silently changing the experiment.

You can override YAML settings from the command line:

```bash
python projects/safe_policy_optimisation/run_experiment.py \
  --pipeline deterministic_minipacman \
  --run-id smoke \
  --total-timesteps 2000
```

The scripts in `stages/` remain available for manual stage-level runs.

### Run output layout

Each launcher run writes a single self-contained directory per stage under
`outputs/<group>/<run_id>/<stage>/` (no extra nesting). Every training stage
produces, in its stage directory:

```text
outputs/deterministic_minipacman/minipacman_default/
  summary.json                     # orchestrator roll-up across stages
  logs/<stage>.log
  ppo_policy/
    model.zip                      # the trained policy
    metrics.json                   # final evaluation: success / reward / safety
    tensorboard/                   # TensorBoard event files
    config.json  summary.json
    episodes.csv  training_episodes.csv  early_stop_evaluations.csv
    learning_curves/
  shielded_policy/ ...             # same layout
  rashomon_policy/ ...
  ppo_lagrangian/                  # safe-RL baselines: <algorithm>.pt per algorithm,
    metrics.json                   #   metrics.json keyed by algorithm
  cpo/ ...
```

`metrics.json` is the standardised final-evaluation artifact (schema in
`utils/metrics.py`): `success` (rate / count vs `success_reward_threshold`),
`reward` (mean / min / max total return), and `safety` (safe-trajectory rate plus
cost-budget violation counts). Reusable cross-run inputs (synthesised shields,
Rashomon sets, rollout GIFs) live separately under `artifacts/`.

## MiniPacman policy optimisation baselines

Train PPO-Lagrangian and PPO-PID-Lagrangian on the MASA-style
`CustomMiniPacman-v0` environment and report cost-constraint violations:

```bash
python projects/safe_policy_optimisation/stages/train_ppo_lagrangian.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}'
```

Train CPO with its separate stage:

```bash
python projects/safe_policy_optimisation/stages/train_cpo.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}'
```

In this MiniPacman example, safety cost is the MASA label-derived
ghost-collision cost on the reached state. An evaluation episode is counted as a
cost-constraint violation when:

```text
episode_cost > cost_limit
```

The default `--cost-limit 0.0` therefore treats any ghost collision as a
violation.

Useful options:

```bash
python projects/safe_policy_optimisation/stages/train_ppo_lagrangian.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}' \
  --algorithms ppo_lagrangian ppo_pid_lagrangian \
  --total-timesteps 10000 \
  --cost-limit 0.0 \
  --eval-episodes 100 \
  --seed 0
```

Artifacts are written to:

```text
projects/safe_policy_optimisation/artifacts/ppo_lagrangian/<run_id>/
```

Each run writes:

- `config.json`: environment, training, and evaluation settings.
- `summary.json`: per-algorithm reward, cost, violation count, and violation percentage.
- `episodes.csv`: post-training evaluation episode reward, cost, length, and violation flag.
- `training_episodes.csv`: completed training exploration episodes with reward, cost, length, end timestep, and violation flag.
- `<algorithm>.pt`: model parameter checkpoint and run metadata.

`summary.json` keeps post-training evaluation metrics under the original flat
keys (`violation_count`, `violation_percentage`) and stores exploration-time
counts with `training_` prefixes (`training_violation_count`,
`training_violation_percentage`).

## Roll out a trained policy to GIF

Generate one animated GIF per rollout episode from a saved checkpoint:

```bash
python projects/safe_policy_optimisation/stages/rollout_policy_gif.py \
  --checkpoint projects/safe_policy_optimisation/artifacts/ppo_lagrangian/<run_id>/ppo_lagrangian.pt \
  --episodes 5
```

Or load from a run directory plus algorithm name:

```bash
python projects/safe_policy_optimisation/stages/rollout_policy_gif.py \
  --run-dir projects/safe_policy_optimisation/artifacts/cpo/<run_id> \
  --algorithm cpo \
  --episodes 5
```

GIFs are saved to `<checkpoint-parent>/rollouts/` by default, alongside rollout
summary artifacts:

- `<algorithm>_episode_000.gif`, one per episode.
- `<algorithm>_rollout_summary.json`.
- `<algorithm>_rollout_episodes.csv`.

## Train with MASA probabilistic shielding

Train an SB3 PPO policy on `CustomMiniPacman-v0` wrapped by MASA's
`ProbShieldWrapperDisc`:

```bash
python projects/safe_policy_optimisation/stages/train_masa_shielded_policy.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}'
```

The default `--safety-tolerance 0.0` uses a zero-risk safety bound. The MASA
wrapper projects augmented policy actions before they reach the environment, so the
policy is trained in the shielded action space.

Useful options:

```bash
python projects/safe_policy_optimisation/stages/train_masa_shielded_policy.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}' \
  --total-timesteps 10000 \
  --eval-episodes 100 \
  --safety-tolerance 0.0 \
  --seed 0
```

Artifacts are written to:

```text
projects/safe_policy_optimisation/artifacts/masa_shielded_policy/<run_id>/
```

Each shielded run writes `model.zip`, `config.json`, `summary.json`,
`training_episodes.csv`, and `episodes.csv`.

## Train with a precomputed shield

For a strict separation between shield synthesis and policy optimisation, train
PPO with an already-saved shield artifact:

```bash
python projects/safe_policy_optimisation/stages/train_discrete_shielded_policy.py \
  --shield-path projects/safe_crl/pipelines/safety_retention/CustomMiniPacman/artifacts/shields/minipacman_default/shield_q.pt \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}' \
  --max-episode-steps 100
```

This script does not synthesise a shield. It loads a binary `(state, action)`
mask from `shield_q.pt`, creates the requested unshielded Gymnasium env, and
uses `ProvablySafePPO` to override unsafe proposed actions during rollout
collection. By default, PPO stores and optimises against the proposed action
(`--shield-action-storage proposed`), while the environment is stepped with the
shielded action. Use `--shield-action-storage executed` to store the overridden
action and recompute its log-probability, matching the previous implementation.

Artifacts are written to:

```text
projects/safe_policy_optimisation/artifacts/shielded_policy/<run_id>/
```

Each run writes `model.zip`, `config.json`, `summary.json`,
`training_episodes.csv`, and `episodes.csv`, including shield intervention
diagnostics.

## Synthesise a shield

Use the project-local shield synthesis entry point to create `shield_q.pt` before
running precomputed-shield policy optimisation:

```bash
python projects/safe_policy_optimisation/stages/synthesise_shield.py \
  --env CustomMiniPacman-v0 \
  --task minipacman_default \
  --max-episode-steps 100 \
  --constraint PCTL \
  --constraint-kwargs '{"alpha": 0.01}' \
  --init-safety-bound 1e-12 \
  --theta 1e-12 \
  --max-vi-steps 2000 \
  --granularity 10
```

This reuses the safety-retention shield synthesis implementation and writes by
default to:

```text
projects/safe_policy_optimisation/artifacts/shields/<env>/<task>/shield_q.pt
```

You can pass `--output-dir` to place the shield elsewhere. The generated
`shield_q.pt` can be passed directly to `train_discrete_shielded_policy.py` via
`--shield-path`. For PCTL constraints, `constraint_kwargs.alpha` is used as the
shield action-risk threshold; when no alpha is provided the threshold is `0.0`.
