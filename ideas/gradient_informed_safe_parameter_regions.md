# Gradient-Informed Safe Parameter Regions for PSPO

## Status and scope

This document describes how policy-gradient and optimizer-state information can guide
the computation of a PSPO safe parameter region. The objective is to allocate more
certified parameter width to coordinates and directions that the policy optimizer is
likely to use.

Gradient information is a utility signal, not a safety argument. The complete region
must still pass the existing PSPO certificate. A poor or unstable update estimate may
produce an inefficient region, but it must not weaken the safety condition.

## 1. Motivation

The current orthotope search maximizes an unweighted combination of total interval width
and log-volume. Before considering the effect on the safety constraint, every parameter
coordinate is treated as equally valuable.

This can waste certification capacity. The policy optimizer may consistently move only
a subset of parameters, or may strongly prefer one side of a parameter while the safe
region allocates width elsewhere.

The desired behaviour is:

1. Estimate the displacement that the policy optimizer wants to make.
2. Assign higher utility to interval width that supports that displacement.
3. Let certificate gradients determine where that requested width is safe or expensive.
4. Accept only a region that passes the unchanged safety certificate.

## 2. Prefer the effective optimizer update over the raw gradient

Let `theta_0` be the current certified-safe parameters and `g_t` the current policy-loss
gradient. For Adam, the approximate update is

```text
m_t     = beta_1 * m_(t-1) + (1 - beta_1) * g_t
v_t     = beta_2 * v_(t-1) + (1 - beta_2) * g_t^2
d_t     = -eta_t * m_hat_t / (sqrt(v_hat_t) + epsilon)
theta_1 = theta_0 + d_t
```

The raw gradient does not include momentum, second-moment normalization, the active
learning rate, or other optimizer behaviour. Consequently, the preferred guidance
signals, in descending order, are:

1. **Realized parameter displacement**

   ```text
   d_realized = theta_candidate - theta_0
   ```

   This includes gradient clipping, momentum, second moments, learning-rate schedules,
   parameter-group learning rates, and weight decay.

2. **Predicted Adam displacement**

   Compute the next update from the current gradient and a read-only copy of the Adam
   state. This is useful when the safe region must be built before applying the candidate.

3. **Raw gradient direction**

   Use `-g_t` only when optimizer state is unavailable. It should be normalized and
   treated as lower-confidence guidance.

For gradient-step adaptive PSPO, the realized displacement is readily available after
the policy optimizer creates a candidate. For train-phase adaptation, the net
displacement over all minibatch steps is available, but it describes a longer and less
local trajectory.

## 3. Asymmetric gradient-informed intervals

Represent each parameter interval around `theta_0` using independent negative and
positive radii:

```text
theta_i in [theta_0_i - r_minus_i, theta_0_i + r_plus_i]
r_minus_i >= 0
r_plus_i  >= 0
```

Given a preferred displacement `d_i`:

- positive `d_i` should prioritize `r_plus_i`;
- negative `d_i` should prioritize `r_minus_i`;
- negligible `d_i` should receive only baseline width unless the certificate optimizer
  finds that width especially inexpensive.

The current bounded model already has independent lower and upper variables, so this
does not require a new geometric region type. It requires a directional objective and,
optionally, directional initialization.

## 4. Guidance objectives

Several objectives can express the desired width allocation. They can be combined with
the existing size objective.

### 4.1 Weighted asymmetric log-width

Define positive and negative utility weights:

```text
w_plus_i  = w_floor + lambda_dir * demand_plus_i
w_minus_i = w_floor + lambda_dir * demand_minus_i
```

where

```text
demand_plus_i  = max(d_i, 0)
demand_minus_i = max(-d_i, 0).
```

Then maximize

```text
J_directional =
    sum_i w_plus_i  * log(r_plus_i  + epsilon)
  + sum_i w_minus_i * log(r_minus_i + epsilon).
```

`w_floor` keeps a minimum utility for flexibility away from the current update. Without
it, coordinates with no current demand may collapse to zero width.

Advantages:

- simple and differentiable;
- directly produces asymmetric intervals;
- integrates with the current constrained optimization.

Limitations:

- the scale and normalization of the weights affect the result;
- it does not explicitly ensure that a common fraction of the candidate update fits.

### 4.2 Projected-update retention

For the current radii, the part of the preferred update contained by the box is

```text
d_box_i = clamp(d_i, -r_minus_i, r_plus_i).
```

Maximize retained update utility, for example

```text
J_retention = -sum_i q_i * (d_i - d_box_i)^2.
```

This directly penalizes width allocations that cause large projection corrections.

Advantages:

- aligned with the behaviour of the projected policy optimizer;
- has a clear diagnostic interpretation;
- can use a metric or per-layer weights `q_i`.

Limitations:

- piecewise gradients can concentrate on currently clipped coordinates;
- fitting coordinates independently may distort the overall update direction.

### 4.3 Common update-ray coverage

The largest common fraction of `d` contained in the box is

```text
alpha_box = min over active i of:
    r_plus_i  / d_i       when d_i > 0
    r_minus_i / (-d_i)    when d_i < 0.
```

Coordinates with `|d_i|` below a threshold should be excluded from the minimum. A smooth
minimum can be used during optimization.

Maximizing `alpha_box` prevents one narrow coordinate from clipping the entire update
ray. It is the most direct answer to the question, "How much of the optimizer step does
this safe box permit?"

Advantages:

- preserves a common scaling of the proposed update;
- naturally supports safe line search;
- gives a useful scalar diagnostic.

Limitations:

- a hard minimum is nonsmooth;
- tiny update coordinates need thresholding;
- an orthotope containing the ray also contains many unrelated corner combinations.

### 4.4 Recommended combined objective

Use

```text
J = J_size
    + lambda_ray * J_ray
    + lambda_retention * J_retention
    + lambda_asymmetry * J_directional.
```

The certificate remains a constrained-optimization condition rather than becoming a
weighted term that can be traded away. Start with `J_size + lambda_ray * J_ray`, then add
the other terms only if diagnostics show a need.

## 5. Width demand from momentum and update history

A single stochastic gradient can be noisy. Momentum and recent updates can estimate the
direction that training is likely to continue using.

### 5.1 One-step Adam demand

Use the magnitude and sign of the predicted or realized Adam displacement. This already
combines the first and second moments into the effective per-coordinate step.

### 5.2 Exponential moving average of realized updates

Maintain

```text
d_ema_t = gamma * d_ema_(t-1) + (1 - gamma) * d_realized_t.
```

This is easy to interpret and includes all optimizer effects. It may be more stable than
reading Adam's moments directly, especially when gradient clipping or weight decay is
active.

### 5.3 Short-horizon momentum trajectory

Predict cumulative displacements over a small horizon `H` using the current optimizer
state and learning-rate schedule. Define side-specific demand as

```text
D_plus_i  = max over k=1..H of max(cumulative_delta_k_i, 0)
D_minus_i = max over k=1..H of max(-cumulative_delta_k_i, 0).
```

This requests enough width for the likely optimizer trajectory rather than only the next
step. Because future gradients are unknown, the prediction should use a short horizon
and be treated as a preference, not a guarantee.

### 5.4 Envelope or quantile of recent updates

Store a short history of update displacements and use positive and negative quantiles.
This supports parameters whose direction changes over time without immediately making
all intervals symmetric.

For example:

```text
D_plus_i  = quantile(max(d_history_i, 0), q)
D_minus_i = quantile(max(-d_history_i, 0), q).
```

### 5.5 Gradient-momentum disagreement

When the current gradient and momentum disagree, the direction estimate is less stable.
Possible responses are:

- blend the one-step update with the update EMA;
- increase the opposite-side width floor;
- reduce `lambda_dir` using an alignment score;
- fall back to symmetric widths when global cosine similarity is very low.

These choices affect efficiency only. Every resulting region still requires full
certification.

## 6. Learning-rate information

A global scalar learning rate changes the magnitude of the desired region but usually
does not change the relative ranking of coordinate widths. Relative demand is affected
by:

- Adam's second-moment denominator;
- parameter-group learning rates;
- layer-wise scaling;
- momentum;
- gradient clipping;
- weight decay;
- update history.

For a one-step region, use the active learning rate. For a short-horizon region, use the
known future schedule when predicting cumulative displacement.

The policy optimizer's effective learning rates should not simply become per-coordinate
learning rates for the Rashomon-bound optimizer. Doing that changes the search dynamics
without clearly defining the desired final region. Adam can also partially normalize
such gradient scaling. Optimizer information should instead define width demand,
initialization, or the region objective.

## 7. Normalize width demand carefully

Raw parameter updates from different layers may have incomparable scales. Useful
normalization options include:

1. **Per-tensor L1 normalization.** Allocate a layer budget across coordinates according
   to absolute update magnitude.
2. **Per-tensor L2 normalization.** Preserve the update direction within each tensor.
3. **Relative parameter update.** Use `|d_i| / (|theta_0_i| + scale)` with a robust layer
   scale for near-zero parameters.
4. **Optimizer-whitened update.** Use the effective Adam displacement directly, then
   normalize only for objective weighting.
5. **Fisher-weighted demand.** Measure update size in an approximate behavioural metric.

The first implementation should use the realized Adam displacement with per-tensor L2
normalization. It is simple and avoids large layers dominating because they contain more
or numerically larger parameters.

Layer-level demand also needs a policy. Options include equal utility per layer or
utility proportional to each layer's proposed update norm. Both should be measured as an
ablation.

## 8. Interaction with certificate sensitivity

The preferred update says where width is useful. The safety certificate says where width
is costly.

The existing constrained Rashomon optimization already differentiates the safety
surrogate with respect to lower and upper bounds. After adding directional utility, the
optimization should naturally allocate width where:

```text
high policy-update demand
and low certificate degradation
```

coincide.

An optional initialization heuristic could estimate one-sided certificate sensitivity:

```text
c_plus_i  = max(0, -d margin / d r_plus_i)
c_minus_i = max(0, -d margin / d r_minus_i)
```

and initialize requested widths approximately according to demand divided by
`sensitivity + epsilon`. This should remain an initialization heuristic because bound
sensitivities are local, noisy, and strongly coupled across parameters.

## 9. Asymmetric initialization and search

The current scalar initial box width can be generalized to tensor-valued positive and
negative initial radii.

A sound search procedure is:

1. Start from the certified point `theta_0`.
2. Compute normalized side-specific demand.
3. Find a safe common step fraction along the candidate ray by bisection.
4. Initialize preferred-side radii to contain that ray fraction.
5. Initialize opposite-side radii using a configurable floor.
6. Optimize the combined size and direction objective under the certificate constraint.
7. Checkpoint and independently certify candidate boxes as currently done.

If initialization creates an uncertified box, shrink toward `theta_0` before starting the
growth optimization. The final guarantee must come from checkpoint certification, not
from the update prediction.

## 10. Orthotope limitation and alternative shapes

An orthotope is axis-aligned. If it contains a dense displacement `alpha * d`, it also
contains all coordinate-wise mixtures between `theta_0` and
`theta_0 + alpha * d`. These mixed corners may not resemble any optimizer trajectory and
may be difficult to certify.

A rank-one zonotope represents the gradient ray directly:

```text
theta = theta_0 + z * d
z in [0, alpha].
```

This avoids unrelated orthotope corners. Additional generators can represent:

- the update EMA;
- variation between gradient and momentum directions;
- principal components of recent safe updates;
- low-rank lateral freedom around the preferred trajectory.

The current zonotope representation already supports asymmetric coefficient intervals,
so a main update generator with coefficient range `[0, alpha]` is geometrically natural.

The practical progression is therefore:

1. Gradient-informed asymmetric orthotope as the smallest implementation change.
2. Gradient-aligned rank-one or low-rank zonotope if orthotope corners dominate
   certificate failures.
3. Cone or pyramid region if lateral width should grow along the update trajectory.

## 11. Optimizer-state consistency after projection

Projecting or reverting policy parameters does not automatically modify Adam's momentum
and second-moment state. The next optimizer step may therefore continue pushing toward a
rejected or clipped candidate.

This matters for both prediction and training dynamics. Potential policies are:

1. Keep Adam state unchanged and treat repeated boundary pressure as expected projected
   optimization.
2. Remove only the outward component of momentum for coordinates active at a bound.
3. Reset momentum for heavily projected coordinates.
4. Snapshot and restore optimizer state together with the last-safe parameters.

Each option changes learning behaviour and should be configured separately from region
construction. At minimum, diagnostics should record whether the predicted next update
continues to point outside the active region.

When optimizer state is retained after projection, future update predictions must use
that retained state. They must not use a stale state snapshot that no longer matches the
actual training process.

## 12. Precomputed versus adaptive PSPO

### Adaptive PSPO

Adaptive PSPO is the best initial target because a current gradient, candidate update,
and optimizer state naturally exist when the region is recomputed.

The adaptive flow can pass

```text
preferred_delta = theta_candidate - theta_last_safe
```

into the Rashomon computation around `theta_last_safe`.

### Precomputed PSPO

A region computed before RL training does not have a representative RL gradient unless
a rollout and policy-loss batch are generated first. Possible guidance sources are:

- gradients from a representative initial rollout buffer;
- an average over several rollout seeds;
- behavioural-cloning gradients;
- task-specific expected update directions;
- no guidance, preserving the current baseline.

A single precomputed gradient-informed box may become irrelevant as training progresses.
Update-history guidance is therefore more appropriate for adaptive recomputation.

## 13. Proposed implementation plan

### Phase 1: Capture and diagnose update demand

1. Capture parameters immediately before and after each policy optimizer step.
2. Record realized actor displacement in projection-parameter order.
3. Read Adam `exp_avg`, `exp_avg_sq`, step count, and parameter-group learning rates for
   comparison with the realized displacement.
4. Compute layer-normalized positive and negative demand tensors.
5. Add diagnostics without changing region computation.

Diagnostics should include:

- raw-gradient versus realized-update cosine similarity;
- momentum versus current-gradient cosine similarity;
- per-layer update norms;
- predicted versus realized update error;
- current box ray coverage;
- retained update norm and cosine after projection.

### Phase 2: Directional orthotope objective

1. Add an optional preferred displacement to the Rashomon computation API.
2. Express lower and upper radii relative to the nominal parameters.
3. Add a smooth common-ray coverage objective.
4. Preserve the existing size objective and unchanged safety constraints.
5. Add a configurable opposite-side width floor.
6. Keep gradient guidance disabled by default.

### Phase 3: Initialization and history

1. Add tensor-valued positive and negative initial radii.
2. Add realized-update EMA and short update history.
3. Support guidance modes such as:

   ```text
   none
   raw_gradient
   adam_predicted_step
   realized_step
   realized_step_ema
   recent_update_quantile
   ```

4. Add a short optional momentum horizon using the known learning-rate schedule.

### Phase 4: Gradient-aligned zonotope comparison

1. Add the effective update as the first zonotope generator.
2. Search an asymmetric coefficient range `[0, alpha]` for that generator.
3. Add a small number of recent-update or correlation generators.
4. Compare certification tightness and policy progress against the guided orthotope.

### Phase 5: Momentum handling at projected boundaries

1. Measure repeated outward updates and projection frequency.
2. Compare retaining, truncating, and restoring momentum state.
3. Keep optimizer-state handling independent of the safety certificate and region-shape
   choice.

## 14. Suggested initial experiment

Compare the following guidance modes under identical seeds and certificate settings:

```text
baseline:              current unweighted safe-box objective
raw-gradient:          side weights from the negative policy gradient
adam-step:             side weights from the realized Adam displacement
adam-step-plus-ray:    realized displacement plus common-ray coverage
update-EMA:            common-ray coverage using an EMA of realized updates
```

Use adaptive PSPO at gradient-step granularity and start with:

```text
normalization:          per-tensor L2
opposite-side floor:    small but nonzero
prediction horizon:     one step
certificate:            unchanged
fallback:               last certified region or last safe policy
```

Measure:

- certified common step fraction `alpha_box`;
- fraction of candidate update norm retained after projection;
- cosine similarity between candidate and projected update;
- certificate success rate;
- projection frequency and displacement;
- positive/negative interval asymmetry;
- region computation time;
- policy return under the same safety specification.

## 15. Safety invariants

1. Gradient and optimizer information may influence only region proposal, initialization,
   or size utility.
2. The safety target and final certificate threshold must remain unchanged.
3. Every installed region must be fully certified after gradient-guided optimization.
4. A preferred candidate endpoint must not be assumed safe merely because the region was
   shaped toward it.
5. Stale or unstable momentum information must not bypass recertification.
6. Guidance tensors must match the certified model's parameter order, shapes, dtype, and
   device.
7. Projection failure must return to a known safe policy.
8. The guarantee retains the scope of the underlying PSPO specification and certificate.

## 16. Recommendation

Start by using the realized Adam displacement to optimize a common safe fraction of the
candidate update, while retaining the existing size objective and certificate. Use
separate positive and negative radii with a small opposite-side floor.

This approach uses all optimizer information indirectly through the update it actually
produced, requires no new verifier geometry, and gives a clear success metric. If the
resulting boxes remain overly conservative because of unrelated axis-aligned corners,
move the same guidance signal into a rank-one or low-rank gradient-aligned zonotope.
