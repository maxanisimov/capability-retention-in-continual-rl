# PSPO policy-initialisation ablation launch

This run launches only `no_entropy` and `ce_only` across the six MASA
environments and seeds 0--9. The canonical control is validated and reused; it
is never rerun by these commands. Production consists of 12 shared base-policy
jobs followed by 120 PPO jobs.

## Invariants

- `no_entropy`: margin-loss weight 1, entropy weight 0, target margin 2 stopping.
- `ce_only`: margin-loss weight 0, entropy weight 0, **any-safe feasibility**
  stopping (`--bc-margin-mode any`), and the Rashomon set is built in the same
  `any` mode. A failed initializer is recorded and is not automatically retried.

  Rationale: `safe_action_bc_loss` is `-log P(safe set)`, a logsumexp over the
  safe actions, so it is governed by the *best* safe logit. The original
  strict all-safe criterion tested the *worst* safe logit -- a quantity the
  loss never optimises -- and four of six environments reached 100%
  allowed-action accuracy yet failed, with margins from -0.10 to -12.29.

  Note this makes `ce_only` differ from `no_entropy` in two respects, not one:
  the initialiser objective *and* the Rashomon multi-label mode. Report the
  comparison accordingly; it is not a single-factor ablation.
- Both variants otherwise retain the canonical directional adaptive LID,
  200-iteration budget, full-state batch and certificate, weighted-width
  objective, replacement mode, PPO settings, horizon, and environment-specific
  rollout frequency.
- Every job gets one exact CPU ID, all numerical-library thread counts are one,
  and the process is run at niceness 5.

## Authentication and clean source

Use the dedicated clean, committed worktree recorded in the dispatch manifest.
Its ignored `.venv` and `projects/safe_policy_optimisation/artifacts` paths may
be symlinks to the main shared checkout. The dispatcher refuses a dirty source
unless `--allow-dirty` is explicitly passed; never use that override for
production.

Start and export one agent in the controlling shell:

```bash
ssh-agent -a /tmp/$USER-pspo-policy-init-agent.sock
export SSH_AUTH_SOCK=/tmp/$USER-pspo-policy-init-agent.sock
ssh-add ~/.ssh/id_ed25519
ssh-add -l
```

Stop if `ssh-add -l` does not list the intended key.

## Targeted gate

From the clean worktree, run:

```bash
.venv/bin/python -m unittest \
  projects.safe_policy_optimisation.tests.test_pspo_initialisation_cluster_unittest \
  projects.safe_policy_optimisation.tests.test_pspo_four_ablations_unittest \
  projects.safe_policy_optimisation.tests.test_pspo_launcher_unittest \
  projects.safe_policy_optimisation.tests.test_pspo_multi_env_launcher_unittest

.venv/bin/python -m unittest \
  core.provably_safe_policy_optimisation.test_adaptive_safe_ppo_unittest
```

The known unrelated legacy project-discovery failures are recorded in the
manifest/run notes and do not replace this gate.

## Exact core probe

Probe after the gate and before each wave. A core qualifies only when its
five-sample average is at least 90% idle. CPUs 0 and 1 are reserved, at least 2
GB of currently available RAM is budgeted per job, and the worktree, virtual
environment, and shared artifact mount must be visible on the host.

```bash
LAUNCH_ROOT=/absolute/path/to/the/clean/worktree
HOSTS_TSV=/tmp/pspo_policy_initialisation_hosts.tsv
cd "$LAUNCH_ROOT"

$LAUNCH_ROOT/.venv/bin/python \
  $LAUNCH_ROOT/projects/safe_policy_optimisation/scripts/lab_cluster/pspo_initialisation.py \
  probe --source-root "$LAUNCH_ROOT" --out "$HOSTS_TSV" \
  --minimum-idle 90 --reserve 2 --samples 5
```

The TSV stores exact, potentially noncontiguous CPU IDs and the probe policy.
Dispatch rejects a TSV generated with a different idle threshold or reserve.

## Wave 1: base policies

Review the plan first, then launch the same plan. All 12 outstanding units must
fit; otherwise production launch aborts without starting a partial wave.

```bash
OUTPUT_ROOT=/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning/artifacts/ablation_studies/pspo_four_ablations
TOOL=$LAUNCH_ROOT/projects/safe_policy_optimisation/scripts/lab_cluster/pspo_initialisation.py

$LAUNCH_ROOT/.venv/bin/python "$TOOL" dispatch \
  --hosts-tsv "$HOSTS_TSV" --source-root "$LAUNCH_ROOT" \
  --output-root "$OUTPUT_ROOT" --phase base

$LAUNCH_ROOT/.venv/bin/python "$TOOL" dispatch \
  --hosts-tsv "$HOSTS_TSV" --source-root "$LAUNCH_ROOT" \
  --output-root "$OUTPUT_ROOT" --phase base --launch
```

Monitor until all 12 report `ready`:

```bash
$LAUNCH_ROOT/.venv/bin/python "$TOOL" status \
  --output-root "$OUTPUT_ROOT" --check-hosts
```

Do not begin Wave 2 if any `INITIALISER_FAILED` marker exists. This preserves the
specified no-retry/no-retuning rule.

## Wave 2: PPO seeds

Re-probe, review, and launch. The dispatcher validates all prepared base
artifacts and all ten seeds of every canonical control before it plans training.
It spreads one job per host on the first pass and uses additional idle CPUs only
if fewer than 120 suitable hosts are available. Immediately before starting any
process it measures every assigned CPU again; if one has become busy, nothing is
launched and the probe must be repeated.

```bash
$LAUNCH_ROOT/.venv/bin/python "$TOOL" probe \
  --source-root "$LAUNCH_ROOT" --out "$HOSTS_TSV" \
  --minimum-idle 90 --reserve 2 --samples 5

$LAUNCH_ROOT/.venv/bin/python "$TOOL" dispatch \
  --hosts-tsv "$HOSTS_TSV" --source-root "$LAUNCH_ROOT" \
  --output-root "$OUTPUT_ROOT" --phase train

$LAUNCH_ROOT/.venv/bin/python "$TOOL" dispatch \
  --hosts-tsv "$HOSTS_TSV" --source-root "$LAUNCH_ROOT" \
  --output-root "$OUTPUT_ROOT" --phase train --launch
```

Each dispatch has its own immutable manifest beneath
`_policy_initialisation_dispatch/<phase>_<UTC>/`, containing the commit, source
hashes, probe hash, exact host/core assignment, canonical source hashes,
prepared-base hashes, and resolved experiment settings. Rerunning a wave is
resumable: valid `metrics.json` seeds are skipped.

## Completion and analysis

Require 12 valid base summaries and 120 valid new seed metrics. Then run only
the policy-initialisation analysis variants:

```bash
$LAUNCH_ROOT/.venv/bin/python \
  $LAUNCH_ROOT/projects/safe_policy_optimisation/scripts/analyse_pspo_four_ablations.py \
  --root "$OUTPUT_ROOT" --variants control no_entropy ce_only \
  --output-dir "$OUTPUT_ROOT/analysis/policy_initialisation"
```

The resulting paired-seed report compares `no_entropy` with control (entropy),
`ce_only` with `no_entropy` (margin), and both variants with control, while
retaining failed or non-attaining seeds.
