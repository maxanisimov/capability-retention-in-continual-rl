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

## Run paper-scale safe-RL and PSPO experiments

Run the commands in this section from the repository root with the virtual
environment activated:

```bash
source .venv/bin/activate
```

The paper launchers recognise these short environment names:

| Short name | Environment |
|---|---|
| `media_streaming` | Media Streaming |
| `colour_bomb` | Colour Bomb v1 |
| `colour_bomb_v2` | Colour Bomb v2 |
| `bridge_crossing` | Bridge Crossing v1 |
| `bridge_crossing_v2` | Bridge Crossing v2 |
| `mini_pacman` | MiniPacman |

Seed lists passed to the Python multi-seed launchers are comma-separated. Each
launcher resolves the environment, PPO, evaluation, and shield settings from
the corresponding `paper_2503_07671_*` pipeline.

### Safe-RL baselines

Use `run_seed_experiments.py` for PPO and the safe-RL baseline methods:

| Method group | Methods |
|---|---|
| `baselines_lag` | PPO-Lagrangian and PPO-PID-Lagrangian |
| `cpo` | CPO |
| `shielded` | PPO-Shield, including nominal-policy evaluation |
| `ppo` | unconstrained PPO reference; not included by default |
| `rashomon` | PSPO precomputed; this is not an adaptive run |

For example, run only the safety-aware RL baselines for Bridge Crossing v1 and
v2 over ten seeds:

```bash
ENVS=bridge_crossing,bridge_crossing_v2 \
SEEDS=0,1,2,3,4,5,6,7,8,9 \
METHOD_GROUPS=baselines_lag,cpo,shielded \
PAPER_OUT_BASE=projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/safe_rl_baselines \
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_seed_experiments.py
```

Add the unconstrained PPO reference with:

```bash
METHOD_GROUPS=ppo,baselines_lag,cpo,shielded
```

Add precomputed PSPO with:

```bash
METHOD_GROUPS=ppo,baselines_lag,cpo,shielded,rashomon
```

Without `METHOD_GROUPS`, the launcher runs `baselines_lag`, `cpo`, `shielded`,
and `rashomon`; it deliberately omits unconstrained PPO. Every method group and
seed is an independent job. The launcher pins jobs to CPU cores and writes logs
under `<PAPER_OUT_BASE>/_launch_logs/`.

Useful baseline launcher controls are:

| Environment variable | Default | Meaning |
|---|---|---|
| `ENVS` | all six | comma-separated environment names |
| `SEEDS` | `0` through `9` | comma-separated seeds |
| `METHOD_GROUPS` | safe baselines plus precomputed PSPO | comma-separated groups from the table above |
| `SWEEP_PARALLEL` | all jobs | maximum concurrent jobs |
| `CPU_OFFSET` | `0` | offset into the available CPU affinity set |
| `NO_PIN` | unset | set to `1` to disable explicit CPU pinning |
| `SMOKE_TIMESTEPS` | unset | replace each training budget for a smoke test |
| `PAPER_OUT_BASE` | `projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/no_earlystop` | result root |

For a quick launcher check, use a fresh output directory:

```bash
ENVS=bridge_crossing \
SEEDS=0 \
METHOD_GROUPS=baselines_lag,cpo,shielded \
SMOKE_TIMESTEPS=2000 \
PAPER_OUT_BASE=/tmp/safe_rl_smoke \
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_seed_experiments.py
```

### PSPO adaptive

`stages/train_pspo_adaptive.py` is the canonical PSPO-adaptive training stage.
For normal paper-scale runs, use `run_adaptive_seed_experiments.py`; it prepares
or reuses the required all-safe, one-hot base policy and invokes the canonical
stage once for every environment and seed.

The default adaptive experiment is region-first, directional, uses the
all-safe-action LogSumExp certificate, replaces the previous certified region,
and enforces every PPO optimizer update:

```bash
ENVS=bridge_crossing_v2 \
SEEDS=0,1,2,3,4,5,6,7,8,9 \
ADAPTIVE_VERIFY_FIRST=false \
ADAPTIVE_FREQ=update \
ADAPTIVE_DIRECTIONAL=true \
ADAPTIVE_REGION_MODE=replace \
ADAPTIVE_SURROGATE=logsumexp \
ADAPTIVE_N_ITERS=100 \
ADAPTIVE_OUT_BASE=projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/pspo_adaptive_bridge_v2 \
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/run_adaptive_seed_experiments.py
```

Adaptive launcher controls are:

| Environment variable | Default | Meaning |
|---|---|---|
| `ENVS` | all six | comma-separated environment names |
| `SEEDS` | `0` through `9` | comma-separated seeds |
| `ADAPTIVE_VERIFY_FIRST` | `false` | `false` for region-first; `true` for verify-then-project |
| `ADAPTIVE_FREQ` | `update` | `update`, `rollout`, `once`, or a positive number of rollouts |
| `ADAPTIVE_DIRECTIONAL` | `true` | grow the orthotope toward the proposed parameter update |
| `ADAPTIVE_REGION_MODE` | `replace` | `replace` or `union` certified regions |
| `ADAPTIVE_SURROGATE` | `logsumexp` | `logsumexp` or `probability` |
| `ADAPTIVE_N_ITERS` | `100` | maximum iterations per region computation |
| `CPU_OFFSET` | `0` | offset into the available CPU affinity set |
| `NO_PIN` | unset | set to `1` to disable one-core-per-seed pinning |
| `SMOKE_TIMESTEPS` | unset | replace each environment's training budget |
| `ADAPTIVE_OUT_BASE` | `projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/pspo_adaptive` | adaptive result root |

Numeric `ADAPTIVE_FREQ=N` aggregates N PPO rollouts before enforcing the
candidate update. `ADAPTIVE_FREQ=once` computes one fixed, non-directional
initial region, so it must be combined with `ADAPTIVE_DIRECTIONAL=false` and
`ADAPTIVE_VERIFY_FIRST=false`. Directional growth and proposal-containment
stopping require an orthotope region. `ADAPTIVE_N_ITERS` is a maximum: growth
can finish earlier once the proposal lies inside a fully certified region.

Results are written to:

```text
<ADAPTIVE_OUT_BASE>/
  _base_policies/<environment>/
  _launch_logs/
  <environment>/seed<seed>/
    config.json  metrics.json  summary.json  model.zip
```

For one environment with explicit CPU allocation and selectable tabular,
one-hidden, or two-hidden architecture, use:

```bash
ENV_NAME=bridge_crossing_v2 \
SEEDS="0 1 2 3 4 5 6 7 8 9" \
CPU_IDS=0-9 \
ARCHITECTURE=two_hidden \
RASHOMON_N_ITERS=200 \
RUN_NAME=pspo_adaptive_bridge_v2_two_hidden \
  projects/safe_policy_optimisation/scripts/run_pspo_adaptive_one_env.sh
```

For simultaneous one- or two-hidden runs across several environments, with
automatic idle-core selection, use:

```bash
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/launch_pspo_adaptive_multi_env.py \
  --architecture two_hidden \
  --rashomon-n-iters 200
```

This specialised multi-environment launcher excludes Media Streaming by
default. Pass `--cpu-ids`, `--envs`, and `--seeds` for an explicit allocation,
or `--dry-run` to inspect every resolved command without starting training.

The lower-level stage remains available for a single run when a compatible
base policy and shield already exist:

```bash
.venv/bin/python \
  projects/safe_policy_optimisation/stages/train_pspo_adaptive.py \
  --base-policy-path PATH/base_policy.pt \
  --shield-path PATH/shield_q.pt \
  --env-id ENV_ID \
  --state-representation one_hot \
  --verify-first false \
  --freq update \
  --directional true \
  --region-mode replace \
  --n-iters 100 \
  --rashomon-multi-label-mode all \
  --surrogate logsumexp
```

See [running_experiments.md](docs/running_experiments.md) for ablations and
additional launcher details.

## Manual MiniPacman policy optimisation baselines

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

## Train with MASA probabilistic shielding (archived)

> This trainer is no longer part of the pipeline and has moved to
> `archive/stages/train_masa_shielded_policy.py`. Its environment builder is
> still live in `utils/masa_env.py`, which is what `stages/rollout_policy_gif.py`
> uses to render MASA rollouts.

Train an SB3 PPO policy on `CustomMiniPacman-v0` wrapped by MASA's
`ProbShieldWrapperDisc`:

```bash
python projects/safe_policy_optimisation/archive/stages/train_masa_shielded_policy.py \
  --env-id CustomMiniPacman-v0 \
  --env-kwargs '{"ghost_rand_prob": 0.0}'
```

The default `--safety-tolerance 0.0` uses a zero-risk safety bound. The MASA
wrapper projects augmented policy actions before they reach the environment, so the
policy is trained in the shielded action space.

Useful options:

```bash
python projects/safe_policy_optimisation/archive/stages/train_masa_shielded_policy.py \
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
python projects/safe_policy_optimisation/stages/train_ppo_shield.py \
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
`shield_q.pt` can be passed directly to `train_ppo_shield.py` via
`--shield-path`. The shield allows only the action(s) achieving the minimum
eventual-unsafe risk in each state (within the value-iteration tolerance
`--theta`) — i.e. the safest policy the environment admits, not an
externally-chosen risk threshold.
