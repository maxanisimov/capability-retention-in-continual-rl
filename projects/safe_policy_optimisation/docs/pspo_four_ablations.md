# PSPO four-ablation suite

The suite is launched by `run_pspo_four_ablations.py`. It validates the ten
canonical paired seeds and shared base-policy hash before real runs, pins one
CPU per seed process, skips completed `metrics.json` artifacts, and stores new
runs below:

```text
artifacts/ablation_studies/pspo_four_ablations/
  <variant>/two_hidden/<environment>/seed<seed>/
```

Render and validate the complete matrix without launching training:

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_pspo_four_ablations.py \
  --dry-run --strict-dry-run --allow-dirty
```

Run one non-result smoke seed for every variant:

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_pspo_four_ablations.py \
  --variants no_entropy ce_only fixed_lid no_gradient \
    region_first_instrumented verify_first \
  --envs media_streaming --seeds 0 --cpu-ids 0 --max-parallel 1 \
  --smoke --allow-dirty \
  --output-root /tmp/pspo_four_ablations_smoke
```

`--smoke` deliberately changes the horizon and PPO rollout to eight steps,
uses two evaluation episodes, and changes each LID budget to one iteration.
Its artifacts must not be used as study results.

Launch the controlled suite after committing the implementation (or pass
`--allow-dirty` only when the recorded dirty repository state is intentional):

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/run_pspo_four_ablations.py \
  --max-parallel 4
```

When `--cpu-ids` is omitted, the launcher allocates from the process's current
CPU affinity. Supply an explicit comma/range list when running under a cluster
scheduler.

Analyse any complete or partial result tree with:

```bash
.venv/bin/python projects/safe_policy_optimisation/scripts/analyse_pspo_four_ablations.py
```

The analysis preserves missing, failed, and non-attaining seeds in the
per-seed outputs. It writes CSV/JSON, a Markdown report, paired seed tables,
reward/safety curves, initializer diagnostics, LID-compute summaries, and the
verify-first false-negative/compute comparison. Exact randomization tests and
bootstrap intervals use seeds—not proposals or episodes—as independent units.
