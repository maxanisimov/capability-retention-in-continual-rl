# Running experiments and ablation studies

This is the CLI reference for the launcher scripts in `scripts/` and
`run_experiment.py` — how to run a single pipeline, sweep it across seeds and
environments, and reproduce the ablation studies. See the top-level
[`README.md`](../README.md) for install/setup and the per-stage output layout;
see [`settings/ablation_studies/README.md`](../settings/ablation_studies/README.md)
for the design rationale behind the two ablations.

All commands below assume the repo root as the working directory and the
project installed editable (`pip install -e .`, see the main README).

```bash
cd /vol/bitbucket/ma5923/_projects/CertifiedContinualLearning
.venv/bin/python projects/safe_policy_optimisation/run_experiment.py --list-pipelines
```

## 1. One pipeline, one run

`run_experiment.py` is the preferred entry point for a single pipeline run. It
resolves settings from `settings/<group>/{pipelines,tasks}.yaml`, synthesises
the shield if needed, and runs every configured stage (baselines, shielded
PPO, PSPO) into one output directory.

```bash
.venv/bin/python projects/safe_policy_optimisation/run_experiment.py \
  --pipeline paper_2503_07671_colour_bomb
```

Useful flags:

| flag | purpose |
|---|---|
| `--list-pipelines` | print every registered pipeline name + description, then exit |
| `--pipeline NAME` | pipeline to run (see `settings/*/pipelines.yaml`) |
| `--task NAME` | override the pipeline's default task |
| `--run-id ID` | override the output run id |
| `--output-dir PATH` | override the output directory |
| `--force-shield-synthesis` | re-synthesise `shield_q.pt` even if it already exists |
| any other flag | passed through to the pipeline stage, overriding the YAML (e.g. `--total-timesteps 2000`) |

```bash
# smoke-test a pipeline fast, without touching the YAML
.venv/bin/python projects/safe_policy_optimisation/run_experiment.py \
  --pipeline deterministic_minipacman --run-id smoke --total-timesteps 2000
```

## 2. Multi-seed sweeps of one or more pipelines

For "run this across N seeds" or "run several pipelines", `run_seed_sweep.py`
wraps `run_experiment.py`: it builds the shield + Rashomon set once per
pipeline (seed 0 runs to completion first), fans the remaining seeds out on
disjoint CPU cores, and aggregates each stage's `metrics.json` to mean ± std
across seeds when done.

```bash
# one pipeline, 5 seeds
.venv/bin/python projects/safe_policy_optimisation/scripts/run_seed_sweep.py \
  --pipeline deterministic_minipacman --n-seeds 5

# several pipelines, explicit seeds, 8 cores each
.venv/bin/python projects/safe_policy_optimisation/scripts/run_seed_sweep.py \
  --pipeline deterministic_minipacman \
  --pipeline deterministic_colour_bomb \
  --seeds 0 1 2 3 --cores-per-job 8

# preview the (pipeline x seed) grid and core-slot plan without launching anything
.venv/bin/python projects/safe_policy_optimisation/scripts/run_seed_sweep.py \
  --pipeline deterministic_minipacman --n-seeds 3 --dry-run
```

Key flags: `--seeds N N N...` / `--n-seeds N` (+ `--base-seed`), `--cores-per-job`,
`--max-parallel`, `--skip-existing` (resume a partial sweep), `--dry-run`.
Args after a bare `--` are forwarded verbatim to every `run_experiment.py` call.

To run several *independent* pipelines concurrently on disjoint cores without
the seed-sweep/aggregation machinery, use `run_parallel_pipelines.py` instead
(same `--pipeline` / `--dry-run` / `-- <extra args>` conventions, plus
`--cores-per-pipeline` and `--reserve-cores`).

## 3. The paper baselines, in parallel (`run_seed_experiments.py`)

This is the launcher that produced `artifacts/paper_2503_07671/runs/no_earlystop/`
— every safe-RL baseline (PPO-Lagrangian, PPO-PID-Lagrangian, CPO), the hard
shield, and precomputed PSPO (`rashomon_policy`), for all 6 paper environments,
10 seeds each. Unlike `run_seed_sweep.py` it splits a seed's *method groups*
across separate processes too (`baselines_lag`, `cpo`, `shielded`, `rashomon`),
so one seed's wall-clock is `max(method)` instead of `sum(method)`.

```bash
python projects/safe_policy_optimisation/scripts/run_seed_experiments.py
```

Configured entirely by environment variables (no argparse CLI):

| env var | default | meaning |
|---|---|---|
| `SEEDS` | `0..9` | comma list of seeds |
| `ENVS` | all 6 | comma list of short env names (`media_streaming`, `colour_bomb`, `colour_bomb_v2`, `bridge_crossing`, `bridge_crossing_v2`, `mini_pacman`) |
| `METHOD_GROUPS` | all 4 | comma list of `baselines_lag,cpo,shielded,rashomon` |
| `SWEEP_PARALLEL` | run everything at once | cap on concurrent jobs |
| `SMOKE_TIMESTEPS` | off | override every method's timestep budget for a fast dry run |
| `SERIAL_METHODS` | off | `=1` restores one-process-per-seed (methods run serially) |
| `PAPER_OUT_BASE` | `artifacts/paper_2503_07671/runs/no_earlystop` | output root |

```bash
# quick smoke test, only colour_bomb, 2 seeds
ENVS=colour_bomb SEEDS=0,1 SMOKE_TIMESTEPS=2000 \
  python projects/safe_policy_optimisation/scripts/run_seed_experiments.py
```

## 4. PSPO adaptive (`run_adaptive_seed_experiments.py`)

PSPO adaptive exposes the former v1 and v2 update rules through one stage and
one result name (`pspo_adaptive`). Region-first behavior is the default; use
`--verify-first true` for verify-then-project behavior.

The canonical stage arguments are:

| argument | default | meaning |
|---|---|---|
| `--verify-first` | `false` | verify a candidate before region synthesis (`true`), or synthesize a region first (`false`) |
| `--freq` | `update` | `update`, `rollout`, `once`, or a positive integer for every N rollouts |
| `--directional` | `true` | grow an orthotope only toward the proposed update |
| `--n-iters` | `100` | maximum iterations for each region computation, including the initial region |
| `--surrogate` | `logsumexp` | `probability` or `logsumexp`; `auto` remains as a legacy compatibility value |
| `--region-mode` | `replace` | replace the previous region or retain a union of certified regions |

Both surrogate forms default to the hard requirement that every safe-action
logit exceed every unsafe-action logit. With numeric `--freq N`, PPO updates
are aggregated for N rollouts under shielded exploration and then enforced.
Any pending aggregate update is enforced before final evaluation and saving.

The strongest-frequency region-first invocation is:

```bash
.venv/bin/python projects/safe_policy_optimisation/stages/train_pspo_adaptive.py \
  --base-policy-path PATH/base_policy.pt \
  --shield-path PATH/shield.pt \
  --env-id ENV_ID
```

To compute one non-directional safe region around the initial policy and use it
for every projected optimizer update:

```bash
.venv/bin/python projects/safe_policy_optimisation/stages/train_pspo_adaptive.py \
  --base-policy-path PATH/base_policy.pt \
  --shield-path PATH/shield.pt \
  --env-id ENV_ID \
  --n-iters 10000 \
  --freq once \
  --directional false \
  --verify-first false
```

The multi-seed launcher pins one CPU core per job:

```bash
python projects/safe_policy_optimisation/scripts/run_adaptive_seed_experiments.py
```

| env var | default | meaning |
|---|---|---|
| `SEEDS` | `0..9` | comma list of seeds |
| `ENVS` | all 6 | comma list of short env names |
| `ADAPTIVE_VERIFY_FIRST` | `false` | unified equivalent of former v1 (`true`) or v2 (`false`) behavior |
| `ADAPTIVE_FREQ` | `update` | canonical enforcement frequency |
| `ADAPTIVE_DIRECTIONAL` | `true` | directional safe-region growth |
| `ADAPTIVE_SURROGATE` | `logsumexp` | all-safe probability or log-sum-exp surrogate |
| `ADAPTIVE_REGION_MODE` | `replace` | certified-region replacement/union behavior |
| `ADAPTIVE_N_ITERS` | `100` | per-computation Rashomon budget |
| `ADAPTIVE_OUT_BASE` | `artifacts/paper_2503_07671/runs/pspo_adaptive` | output root |
| `SMOKE_TIMESTEPS` | off | override every env's timestep budget |
| `NO_PIN` | off | `=1` disables per-job `taskset` core pinning |
| `CPU_OFFSET` | `0` | shift core pinning so this launcher can share a machine with another concurrent sweep |

The one-environment launcher invokes the unified stage and additionally accepts:

| env var | default | meaning |
|---|---|---|
| `BC_TARGET_MARGIN` | selected historical setting | required base-policy margin; also controls the direct tabular initialiser |
| `RASHOMON_MULTI_LABEL_MODE` | `all` | hard safe-action-logit semantics; `any` is legacy |
| `RASHOMON_SURROGATE` | `logsumexp` | `probability`, `logsumexp`, or legacy `auto` |
| `RASHOMON_BATCH_SIZE` | selected historical setting | positive integer, or `all` for the complete safety-demonstration dataset |
| `RASHOMON_CERTIFICATE_SAMPLES` | selected historical setting | positive integer, or `all` for every shield state having a safe action |
| `RASHOMON_N_ITERS` | selected setting | maximum iterations for every safe-region computation |
| `ADAPTIVE_FREQ` | `update` | unified enforcement frequency |
| `DIRECTIONAL_RASHOMON_GROWTH` | `1` | `=0` disables proposal-directed orthotope growth |
| `CPU_IDS` | unset | explicit comma-separated, whitespace-separated, or ranged CPU allocation; supersedes `CORE_START` |

When either Rashomon sample setting is `all`, the launcher derives the actual
number of rows from the safety-demonstration dataset. Directional PSPO adaptive
uses a validated base-policy-only artifact: it does not waste an undirected
preliminary Rashomon computation before the first policy proposal is known.
Compatible base-policy artifacts and completed seed runs are reused.

```bash
DIRECTIONAL_RASHOMON_GROWTH=1 \
STOP_WHEN_PROPOSAL_CONTAINED=1 \
ENV_NAME=media_streaming \
CORE_START=0 \
SEEDS="0" \
REGION_MODE=union \
RUN_NAME=pspo_adaptive_media_streaming_directional \
  projects/safe_policy_optimisation/scripts/run_pspo_adaptive_one_env.sh
```

The following launcher runs the `replace`, directional,
all-safe-logit/log-sum-exp experiment for all MASA environments except Media
Streaming. It defaults to two hidden layers; pass `--architecture one_hidden`
for the corresponding one-hidden-layer experiment. Both the optimisation batch
and certificate coverage equal the full safety-demonstration dataset, and 200
iterations are available to every region computation. By default it samples CPU
utilisation for five seconds and launches only if 50 distinct cores are at least
90% idle:

```bash
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/launch_pspo_adaptive_multi_env.py
```

The one-hidden-layer command is:

```bash
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/launch_pspo_adaptive_multi_env.py \
  --architecture one_hidden
```

Inspect the resolved datasets, commands, and allocation without starting jobs:

```bash
.venv/bin/python \
  projects/safe_policy_optimisation/scripts/launch_pspo_adaptive_multi_env.py \
  --cpu-ids 0-49 \
  --dry-run
```

Set `DRY_RUN=1` to print the generated command without launching training.
Directional growth is supported only for orthotope regions. Region growth stops
early when the proposal is contained in a fully certified region, so
`--n-iters` is always a maximum rather than a required spend.

## 5. Ablation studies

Both ablations reuse the `paper_2503_07671` pipeline settings unchanged — only
the ablated variable moves — and land under `artifacts/ablation_studies/`.
Full rationale: [`settings/ablation_studies/README.md`](../settings/ablation_studies/README.md).

### 5.1 Rashomon-set size

How PSPO depends on the optimisation budget used to synthesise the Rashomon
set (baseline `rashomon_n_iters: 2000`, extended to 10k/20k/50k/100k).
`run_rashomon_iter_sweep.py` trains only the `rashomon_policy` stage, and
overrides both `--rashomon-n-iters` and `--rashomon-dir` (a fresh set
directory per iteration count is required, or the pipeline silently reuses
the cached 2k set):

```bash
RASHOMON_N_ITERS=20000 ITER_TAG=rashomon20k \
  python projects/safe_policy_optimisation/scripts/run_rashomon_iter_sweep.py
```

| env var | required? | meaning |
|---|---|---|
| `RASHOMON_N_ITERS` | yes | e.g. `20000` |
| `ITER_TAG` | yes | e.g. `rashomon20k` — becomes the artifact path stem |
| `SEEDS` | no (default `0..9`) | comma list; `SEEDS[0]` is the per-env warmup seed that builds the set before the rest fan out |
| `ENVS` | no (default all 6) | comma list of short env names |
| `SWEEP_PARALLEL` | no (default `20`) | max concurrent jobs |

The pre-built 10k point is also registered as a standalone pipeline set (it's
the only place `colour_bomb_v2` has a 10k point):

```bash
.venv/bin/python projects/safe_policy_optimisation/run_experiment.py \
  --pipelines-file settings/ablation_studies/rashomon_iters/pipelines.yaml \
  --pipeline <name>
```

### 5.2 BC init without projection ("what does the projection buy?")

Keeps the BC-initialised base policy and shielded exploration, but never
projects or reverts an unsafe candidate — every update is still verified, and
the run reports `safe_update_fraction` (how often the policy would have stayed
safe on its own). **This mode gives no safety guarantee**; it exists purely as
a measurement baseline. It needs no pipeline file — it's `run_adaptive_seed_experiments.py`
again, with the ablation strategy selected explicitly:

```bash
ADAPTIVE_STRATEGY=none \
ADAPTIVE_OUT_BASE=projects/safe_policy_optimisation/artifacts/ablation_studies/bc_no_projection \
  python projects/safe_policy_optimisation/scripts/run_adaptive_seed_experiments.py
```

### 5.3 BC-margin sweep (masa_other + media_streaming)

A third ablation, not yet written up in `settings/ablation_studies/`: how the
BC-fit base policy's required logit margin trades off against the reward PSPO
can subsequently recover, comparing the precomputed and adaptive-projection
PSPO variants at each margin. `run_masa_bc_margin_sweep.py` builds a
margin-tagged Rashomon set (`compute_shield_rashomon_set.py --bc-target-margin`)
and then runs both `train_pspo_precomputed.py` and `train_pspo_adaptive.py`
(always with `--unsafe-update-strategy rashomon_project`, the only certified
strategy available) against it, for every margin.

```bash
MARGINS=0.1,1,2,5,10 SEEDS=0,1,2,3,4,5,6,7,8,9 \
  python projects/safe_policy_optimisation/scripts/run_masa_bc_margin_sweep.py
```

| env var | default | meaning |
|---|---|---|
| `MARGINS` | `0.1,1,2,5,10` | comma list of BC logit margins |
| `SEEDS` | `0..9` | comma list of seeds |
| `ENVS` | all 5 masa_other envs | `colour_bomb,colour_bomb_v2,bridge_crossing,bridge_crossing_v2,mini_pacman` (**not** `media_streaming` — that env has its own, separately-run margin sweep already landed under `artifacts/bc_margin_sweep/media_streaming/`) |
| `PHASES` | `sets,precomputed,adaptive_project` | subset of `sets` (build the margin-tagged Rashomon sets), `precomputed`, `adaptive_project` |
| `SET_PARALLEL` | `5` | max concurrent Rashomon-set builds |
| `PSPO_PARALLEL` | `20` | max concurrent precomputed/adaptive training jobs |
| `ADAPTIVE_N_ITERS` | `100` | per-computation Rashomon budget for the adaptive phase |
| `FORCE` | off | `=1` reruns jobs whose output already exists (default: resumable, skips completed jobs) |
| `MASA_MARGIN_OUT_BASE` | `artifacts/bc_margin_sweep/masa_other` | output root |

The launcher is resumable by default (each job checks for its terminal
artifact and skips if present), so re-running the same command after an
interruption picks up where it left off rather than redoing finished seeds.

```bash
# just the margin=0.1 and margin=10 points, colour_bomb only, resume-friendly
MARGINS=0.1,10 ENVS=colour_bomb \
  python projects/safe_policy_optimisation/scripts/run_masa_bc_margin_sweep.py
```

### 5.4 PSPO hyperparameter sweep

`run_pspo_hparam_sweep.py` sweeps only PSPO-specific hyperparameters:
Rashomon optimisation iterations and the BC base-policy logit margin. Each
hyperparameter setting gets a disjoint CPU-core slot; seeds for that setting run
sequentially within the slot, so concurrent settings do not share cores.

```bash
python projects/safe_policy_optimisation/scripts/run_pspo_hparam_sweep.py \
  --env mini_pacman \
  --method precomputed \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --rashomon-iters 100 500 2000 10000 \
  --bc-target-margins 0.1 0.5 1 2 5 \
  --n-hidden 0 \
  --state-representation one_hot \
  --sweep-root outputs/_pspo_hparam/minipacman_tabular
```

Use `--method adaptive` to sweep adaptive PSPO. For precomputed PSPO,
`--rashomon-iters` builds the fixed offline Rashomon box; for adaptive PSPO, it
is the per-update/on-demand Rashomon budget. `rashomon_batch_size` is not swept:
the launcher fixes it to the full safety-demonstration dataset size.

## 6. Plotting / analysis

`plot_unshielded_learning_curves.py` aggregates the per-seed
`unshielded_reward_evaluations` learning curves into one figure per
environment (safety rate, success rate, reward — mean ± N standard errors
across seeds):

```bash
python projects/safe_policy_optimisation/scripts/plot_unshielded_learning_curves.py \
  --root artifacts/paper_2503_07671/runs/no_earlystop --env media_streaming
```

Use `--env NAME` for one environment or `--envs a b c` for several; `--method`
(repeatable) to restrict which methods are drawn instead of the default set;
`--ci-multiplier` to change the standard-error band width (default 2.0).

## Quick reference

| script | what it runs | CLI style |
|---|---|---|
| `run_experiment.py` | one pipeline, one run | argparse |
| `scripts/run_seed_sweep.py` | one or more pipelines × N seeds, aggregated | argparse |
| `scripts/run_parallel_pipelines.py` | several independent pipelines, disjoint cores | argparse |
| `scripts/run_seed_experiments.py` | paper baselines (safe-RL, shield, precomputed PSPO), all envs/seeds | env vars |
| `scripts/run_adaptive_seed_experiments.py` | adaptive PSPO (`AdaptiveSafePPO`), all envs/seeds | env vars |
| `scripts/run_rashomon_iter_sweep.py` | Rashomon-set-size ablation | env vars |
| `scripts/run_masa_bc_margin_sweep.py` | BC-margin ablation (masa_other envs) | env vars |
| `scripts/run_pspo_hparam_sweep.py` | PSPO-only iteration × BC-margin sweep with core-isolated settings | argparse |
| `scripts/plot_unshielded_learning_curves.py` | learning-curve figures from run outputs | argparse |
