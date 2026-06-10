# Experiment Pipelines

Pipeline code is organized by experiment intent.

## Canonical Families

- `behaviour_retention/`: experiments that preserve source-task behaviour from source-policy trajectory data.
- `safety/`: experiments that use shielding to synthesize safe-behaviour datasets.

Hybrid or constrained retention variants stay under `behaviour_retention/` with explicit names, such as `frozenlake_safety_constrained`. The `safety/` family is reserved for shield-generated safety datasets.

## Compatibility Paths

Older top-level packages such as `frozenlake`, `lunarlander`, `frozenlake_safety`, `frozenlake_shield_safety`, and `frozenlake_slippery_shield_safety` are compatibility delegates. New imports and commands should use the canonical family path:

```bash
python -m experiments.pipelines.behaviour_retention.frozenlake.run_experiment
python -m experiments.pipelines.safety.frozenlake.run_experiment
```

Generated run data belongs in pipeline-local `artifacts/` or `outputs/` directories and is ignored by git.
