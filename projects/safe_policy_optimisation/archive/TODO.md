# safe_policy_optimisation — TODO

1. **Fair comparison between PSPO (precomputed) and PSPO (adaptive) on Rashomon
   computation budget.** Precomputed spends a fixed 2,000 iterations on one
   upfront box; adaptive spends 100 iterations per correction, and the number
   of corrections varies per seed (observed 52-269 on the media_streaming
   1-hidden sweep), i.e. 5,200-26,900 total iterations - up to ~13x more than
   precomputed. Before attributing adaptive's reward advantage to the
   *adaptive* mechanism itself, control for this budget confound. Options,
   roughly in order of how directly they isolate adaptivity as the cause:
   - Equalize total budget: rerun precomputed's offline box build with
     `--rashomon-n-iters` set to match adaptive's total (e.g. ~27,000, using
     the max across seeds). If precomputed still can't escape the reward
     floor, that's strong evidence the gap isn't just raw compute.
   - Match iterations-per-decision *and* attempt count: run the offline
     box-builder N times (N = that seed's `rashomon_computations`) with the
     same 100-iteration per-call budget adaptive uses, but always centered on
     the original BC policy (never moving). Train against the best resulting
     box. If repeated independent attempts from the fixed starting point still
     cluster near the floor, that isolates *recentering* (not more tries, not
     more compute) as the active ingredient - the sharpest test of the two.
   - Report an iteration-efficiency frontier instead of one matched point:
     sweep precomputed's `--rashomon-n-iters` (2000, 5000, 10000, ~15000,
     ~27000) and plot reward vs. total iterations spent, with adaptive's
     per-seed points overlaid. Most complete/defensible way to present "value
     of adaptivity" in a paper - shows precomputed's whole achievable
     frontier rather than a cherry-picked budget.
   - Also worth checking before picking a currency: whether iterations are
     even an apples-to-apples unit between the two methods, or whether
     wall-clock time (`rashomon_wall_time_total_s`, already logged for
     adaptive) is fairer if per-iteration cost differs between a global
     462-state box-growth step and a local on-demand one.

2. **Systematic way to choose the required logit margin for BC policy
   initialisation.** Right now the target margin passed to
   `--bc-target-margin` is picked by trial and error, not derived from
   anything about the environment, shield, or architecture - we have no
   principled way to predict ahead of time which margin will leave a
   certifiable Rashomon set. Evidence so far, all for media_streaming, is
   just a grid of empirical outcomes, not a rule:
   - margin=0.1 fails to certify universally - confirmed across 2-hidden
     (default features and one-hot representation), tabular, and both IBP-
     and CROWN-driven growth. Every combination tried collapses to
     `min_hard_acc=0.0` on all checkpoints.
   - margin=2 works well for tabular + one-hot (large certified box, PSPO
     reward close to the reward-optimal baseline) but only escapes the
     reward floor at that specific architecture - precomputed PSPO still
     hits the floor at 1-hidden and 2-hidden even at margin=2.
   - margin=10 (the historical default) is what most of the paper's
     existing results were built on, without ever being explicitly chosen
     for a reason beyond "matches the closed-form linear initialiser's
     default."
   Worth investigating: (a) whether margin sensitivity can be predicted from
   something cheaper to compute than a full Rashomon-set build - e.g. the
   nominal (zero-box) IBP certificate's slack, or a quick small-iteration
   probe; (b) an automatic search procedure (e.g. binary search over margin,
   building a small/fast Rashomon set at each candidate) that finds the
   smallest certifiable margin for a given architecture instead of a human
   guessing; (c) whether the margin/architecture interaction is closer to a
   sharp threshold (as depth vs. tabular already looks like it is) or a
   smooth tradeoff, which would change what "systematic" search should even
   optimise for.
