# Experiments Guide

This directory contains the reproducible experiment pipelines used for the paper release.

## Active Pipelines

### `frozen_lake/`
Main entry points:
- `run_train_and_adapt.py` (single seed)
- `run_train_and_adapt_multi_seed.py` (multi-seed, default seeds `0..9`)
- `postprocessing/` scripts for aggregation and figure/table generation

### `poisoned_apple/`
Main entry points:
- `run_train_and_adapt.py` (single seed)
- `run_train_and_adapt_multi_seed.py` (multi-seed, default seeds `0..9`)
- `postprocessing/aggregate_metrics_across_seeds.py`

### `visualisations/`
Standalone helper scripts for generating environment frames and configuration visuals.

## Output Convention
Each environment follows:

```text
<env>/outputs/<cfg>/<seed>/
├── source/
├── downstream/
├── plots/
└── results_table.csv
```

Logs are stored in:

```text
<env>/logs/<cfg>/<seed>/...
```
