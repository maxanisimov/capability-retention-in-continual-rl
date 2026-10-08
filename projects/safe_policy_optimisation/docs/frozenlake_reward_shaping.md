# FrozenLake training-only potential shaping

Implementation: `utils/frozen_lake_reward_shaping.py`.
PPO test runner: `scripts/run_frozenlake_ppo_shaping.py`.

The opt-in Gymnasium wrapper returns
`r_train = r_raw + scale * (gamma * Phi(next_state) - Phi(state))`.
`Phi` is `1 - d / D` for reachable nonterminal cells, where `d` is the reverse
BFS distance to the goal through non-hole grid cells and `D` is the maximum
finite distance. Goals, holes, and unreachable cells have zero potential.
This is a geometry heuristic, not an optimal policy or expected stochastic
hitting-time computation. It uses no shield mask and costs remain unchanged.

Only training uses the wrapper. Evaluation uses the original environment and
reports goal attainment from the environment's success flag, never a shaped
reward threshold. Raw reward remains `goal_indicator - 0.001 * episode_length`
by default. The step penalty is not removed in this shaping-only experiment.
The PPO actor is randomly initialized with no warm start, action mask, witness
policy, or goal-dependent parameter fitting. The existing PSPO safety-only
initializer is not modified and never consumes this potential. When using this
wrapper with PSPO, construct and audit its initial actor first, then wrap only
the RL training environment; do not pass the potential to initialization.

## Episode-boundary conventions

- Goals and holes: zero next potential; retain the terminal shaping correction.
- `--timeout-mode bootstrap` (default): retain the actual next-state potential
  at a time-limit truncation and retain the truncated flag. SB3 PPO adds the
  discounted terminal value, learned in shaped coordinates, so the residual
  potential cancels. A truncated sample is not treated as a true terminal state.
- `--timeout-mode terminal`: zero next potential at a timeout and turn it into
  a true terminated transition, disabling SB3's time-limit value bootstrap.
  This is a different finite-horizon task convention. Match it in the control.

Use the learner's exact gamma. The training CSV checks the telescoping identity
`G_shaped - G_raw = scale * (gamma**T * Phi_end - Phi_start)` episode by episode.
Potential shaping preserves the underlying discounted objective with these
boundary conventions; it does not fix objectives that reward early failure
more highly than sufficiently slow successful trajectories.

## Reproduce a one-seed 16x16 comparison

Run these commands with a new output directory; existing run IDs are protected
against overwriting. They may run concurrently in separate processes.

```bash
PYTHONPATH=core:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_shaping.py \
  --size 16 --seed 0 --total-timesteps 200000 --shaping-scale 1 \
  --output-dir projects/safe_policy_optimisation/artifacts/frozenlake_shaping_example \
  --run-id shaped_seed0

PYTHONPATH=core:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python projects/safe_policy_optimisation/scripts/run_frozenlake_ppo_shaping.py \
  --size 16 --seed 0 --total-timesteps 200000 --shaping-scale 0 \
  --output-dir projects/safe_policy_optimisation/artifacts/frozenlake_shaping_example \
  --run-id unshaped_seed0
```

Both runs use stochastic transitions (intended direction 0.8, perpendicular
directions 0.1 each), 625-step episode caps, gamma 0.999, two 64-unit tanh layers,
rollout length 2048, 10 optimization epochs, no entropy bonus, no early stopping,
and 100 greedy, unshielded final evaluation episodes. SB3 rounds the requested
200,000 steps up to a complete rollout (200,704 actual steps). Both initial
parameter hashes are recorded so paired initialization can be checked.

Artifacts include `config.json`, source hashes/snapshot, `potential.npz`,
`training_episodes.csv` (raw/shaped and discounted returns, goals, safety, and
telescoping errors), raw evaluation curves, `episodes.csv`, `initial_metrics.json`,
`metrics.json`, `summary.json`, and `model.zip`. Timing distinguishes learning
including periodic evaluation, learning excluding periodic evaluation, and final
evaluation. This dedicated runner leaves source-locked live sweeps unchanged.

References: [Ng et al., ICML 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf);
[Grzes, AAMAS 2017](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf).

## Completed 16x16 pilot (2026-10-07)

Seed 0, 200,704 actual training steps, 100 raw greedy evaluation episodes:

| PPO variant | Raw total reward | Safety | Goal |
| --- | ---: | ---: | ---: |
| Potential shaping, scale 1 | 0.94173 | 98% | 98% |
| Matched unshaped control, scale 0 | 0.86749 | 91% | 91% |

The first successful training trajectory occurred at step 6,768 with shaping
versus 29,266 without it. Initialization parameter hashes were identical.
Training time excluding periodic evaluation was 316.51 s with shaping versus
186.04 s without it in this concurrent pilot: earlier learning in environment
steps did not translate into a shorter full run here. One seed cannot establish
across-seed uncertainty or scalability to larger maps.

Full audited results and artifacts:
[pilot report](../artifacts/paper_2503_07671/runs/ppo_shaping_frozenlake16_20261007T220300Z/README.md).
