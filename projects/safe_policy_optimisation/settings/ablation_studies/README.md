# Ablation-study settings

Two ablations of the PSPO method. Both reuse the `paper_2503_07671` pipelines so
that every non-ablated setting (env kwargs, PPO hyperparameters, budgets,
evaluation) is identical to the main results and only the ablated variable moves.

Results land under `artifacts/ablation_studies/`.

## 1. Rashomon-set size — `rashomon_iters/pipelines.yaml`

How the PSPO results depend on the optimisation budget used to synthesise the
Rashomon set. The baseline is `rashomon_n_iters: 2000` (the value in
`settings/paper_2503_07671/`); this ablation extends it to 10k / 20k / 50k / 100k.

Driven by `scripts/run_rashomon_iter_sweep.py`, which overrides
`--rashomon-n-iters` **and** `--rashomon-dir` per run — a fresh set directory per
iteration count is required, because the pipeline reuses a cached
`rashomon_param_bounds.pt` if one is present and would otherwise silently train
against the 2000-iteration set:

```bash
RASHOMON_N_ITERS=20000 ITER_TAG=rashomon20k \
  python projects/safe_policy_optimisation/scripts/run_rashomon_iter_sweep.py
```

`rashomon_iters/pipelines.yaml` holds the 10k variant as an explicit,
self-contained pipeline set (it also carries the only `colour_bomb_v2` entry, so
that env has a 10k point). Select it with
`--pipelines settings/ablation_studies/rashomon_iters/pipelines.yaml`.

## 2. BC init without projection

What does the projection actually buy? This variant keeps everything else about
PSPO — the behaviourally-cloned base policy fitted on the shield-induced
safe-action demonstrations, and shielded exploration — but **never projects and
never reverts**. Each candidate update is still verified, so the run reports:

* `safe_update_fraction` — the share of policy updates that stayed shield-safe on
  their own, i.e. how often the projection would not have been needed;
* `accepted_unsafe` — updates kept despite failing verification;
* the usual final safety and return metrics, evaluated unshielded.

**This configuration provides no safety guarantee** and exists only as a
measurement baseline.

It needs no pipeline file: it is the adaptive stage with
`--unsafe-update-strategy none`, driven by the existing launcher:

```bash
ADAPTIVE_STRATEGY=none \
ADAPTIVE_OUT_BASE=projects/safe_policy_optimisation/artifacts/ablation_studies/bc_no_projection \
  python projects/safe_policy_optimisation/scripts/run_adaptive_seed_experiments.py
```

The base policy is the BC fit already produced by
`stages/compute_shield_rashomon_set.py` (`fit_base_policy`), read from each env's
`rashomon_fullcov/base_policy.pt`; the Rashomon bounds in that directory are
simply not consulted in this mode.
