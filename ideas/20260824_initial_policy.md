# Initial Policy Quality and Reward Improvement in Adaptive PSPO

## Status and scope

This document collects ideas for assessing whether a safe initial policy is a useful
starting point for reward optimization in adaptive PSPO. It also proposes alternative
ways to construct initial policies when rewards are unavailable before RL training.

The central distinction is between:

1. **Safety confidence:** how strongly the policy currently separates safe and unsafe
   actions.
2. **Safe optimization mobility:** how far, and in how many useful directions, policy
   parameters can move without losing the safety certificate.

The second property is more closely related to whether adaptive PSPO can discover a
high-reward policy. None of the proposed metrics replaces exact safety verification.
They are reward-free utility diagnostics for choosing among policies that satisfy the
same safety requirement.

## 1. Fundamental limitation

Without reward information, or a prior over possible reward functions, it is impossible
to determine which safe initial policy is closest to the reward-optimal policy. Two
environments can have identical dynamics and safety specifications but assign opposite
rewards to two safe actions. No reward-free initialization score can distinguish them.

Consequently, a reward-free notion of a good initialization should not claim to predict
the optimal action. It should instead prefer policies that:

- are exactly safe;
- possess a sufficiently robust safety buffer;
- can move safely in many parameter-space directions;
- can readily change preferences among safe actions;
- are well conditioned for policy-gradient optimization; and
- do not encode an arbitrary, overly strong preference among actions that are all known
  to be safe.

In this sense, a bad initial policy is not necessarily one with a small logit margin. It
is one whose safely reachable neighbourhood is small, narrow, poorly conditioned, or
unable to express alternative safe behaviours.

## 2. Why the minimum logit margin is insufficient

For a state `s`, safe action `a`, unsafe action `u`, parameters `theta`, and action logit
`z`, define the pairwise safety gap

```text
g_(s,a,u)(theta) = z_a(s; theta) - z_u(s; theta).
```

The existing all-safe-actions margin is approximately the minimum of these gaps over
all relevant states and action pairs. This metric is useful for checking functional
separation, but it has several limitations as a predictor of final reward:

1. **It is a single worst-case value.** It discards the distribution of margins across
   states and does not reveal whether one exceptional state or many states are close to
   the boundary.
2. **It is measured in output space.** PSPO projects and certifies changes in parameter
   space. The same output margin can correspond to very different safe parameter
   radii.
3. **It ignores sensitivity.** A gap of two is fragile when a small parameter change
   changes the gap rapidly, but robust when the corresponding Jacobian is small.
4. **It ignores safe-versus-safe preferences.** A policy can give one arbitrary safe
   action nearly all probability and still have an excellent safe-unsafe margin. Such a
   policy may be difficult to redirect when reward becomes available.
5. **It ignores network conditioning.** Large weights and saturated Tanh units can
   produce large logits while giving poor gradients and loose interval bounds.
6. **It is not a complete description of the certified region.** Two policies with
   similar minimum margins can have very different box widths, effective dimensions,
   and directional bottlenecks.
7. **Absolute logits are not identifiable.** Adding the same constant to every logit
   leaves the policy unchanged. Only logit differences or probabilities are meaningful.

The observed PSPO results are consistent with this limitation: initial policies with
very similar achieved margins have produced substantially different final rewards.

## 3. Reward-free metrics for initialization quality

### 3.1 Distribution of functional safety gaps

The minimum gap should be retained as a safety diagnostic, but accompanied by:

- the first and fifth percentiles;
- the median and mean;
- the fraction of constraints within a small distance of zero;
- per-state minimum gaps; and
- separate summaries for different state classes or shield decisions.

This distinguishes one isolated bottleneck from widespread fragility. It is inexpensive
but remains an output-space diagnostic, so it should not be the primary selection
criterion.

### 3.2 Linearized parameter-space safety radius

For an `L-infinity` parameter perturbation, a first-order estimate of the distance to a
pairwise safety boundary is

```text
rho_inf(s, a, u) =
    g_(s,a,u)(theta) / (||grad_theta g_(s,a,u)(theta)||_1 + epsilon).
```

The dual norm in the denominator converts a functional gap into an approximate
parameter-space radius. For an `L2` perturbation, use the `L2` norm of the gradient.
The `L-infinity` version is especially relevant to the axis-aligned orthotopes used by
PSPO.

Raw parameter units should not be mixed indiscriminately. Define a relative scale
`c_i`, for example from the parameter magnitude and a per-layer floor, and evaluate

```text
rho_rel(s, a, u) =
    g_(s,a,u)(theta)
    / (||c * grad_theta g_(s,a,u)(theta)||_1 + epsilon).
```

Recommended summaries are the minimum, fifth percentile, and median radius, together
with the states and layers responsible for the smallest values. This metric directly
explains why a smaller raw margin can nevertheless permit larger safe optimizer steps.

### 3.3 Certified parameter-space radius

The linearized radius may be inaccurate for a nonlinear network. A stronger metric is
the largest relative radius `r` for which interval verification proves safety for every
parameter vector in

```text
theta_i' in [theta_i - r * c_i, theta_i + r * c_i].
```

It can be estimated by binary search. This requires several verifier calls, but only
once per candidate initialization and not throughout full RL training. It is therefore
a practical screening test.

A policy with a slightly smaller output margin but a larger certified relative radius
is normally the more promising initialization for adaptive PSPO.

### 3.4 Directional safe mobility

A symmetric ball is conservative because policy optimization does not need equal
freedom in every direction. For a candidate direction `v`, the linearized distance to
the first violated constraint is

```text
rho(v) = min over constraints i with grad(g_i)^T v < 0 of
         g_i / (-grad(g_i)^T v).
```

Directions can be sampled in several ways:

- isotropic relative parameter directions;
- layer-balanced directions;
- Fisher-whitened directions, which better resemble policy-gradient geometry; or
- pseudo-update directions that increase the probability of particular safe actions.

Useful summaries include:

- the median and fifth-percentile value of `rho(v)`;
- the fraction of directions that permit a PPO-sized step;
- the effective solid angle of the safe tangent cone; and
- the ratio between the largest and smallest nontrivial directional radii.

This measures the probability that an as-yet-unknown reward gradient will point in a
direction that PSPO can follow without substantial projection.

### 3.5 Initial certified-region geometry

Constructing one PSPO region around each candidate base policy gives a diagnostic that
is closely aligned with the actual algorithm. Record:

- relative log-volume;
- minimum, fifth-percentile, and median relative width;
- the number of parameters with effectively zero width;
- width anisotropy or condition number;
- positive versus negative widths;
- the layer producing the bottleneck; and
- the fraction of representative update directions contained by the region.

One initial region computation is much cheaper than repeatedly adapting regions during
a full experiment. It can identify initializations that are safe but lie in a thin or
poorly oriented part of the safe parameter set.

### 3.6 Safe-action neutrality and entropy

Before observing reward, there is generally no basis for choosing one safe action over
another. For each state, calculate the policy conditioned on the safe action set and
measure:

- conditional safe-action entropy;
- KL divergence from a uniform distribution over safe actions;
- variance or range of safe-action logits;
- minimum probability assigned to any safe action; and
- effective number of represented safe actions.

High safe-unsafe separation and high conditional entropy are compatible: unsafe actions
can receive very little total probability while the remaining probability is distributed
approximately uniformly across safe actions.

This is preferable to an initialization that strongly commits to one arbitrary safe
action. After rewards are observed, PPO can break the safe-action symmetry in favour of
rewarding behaviour.

### 3.7 Safe-action controllability

For each state and each safe action, estimate the smallest parameter change required to
make that action preferred while retaining safety at all shield states. This tests
whether the network can express alternative safe behaviours.

Related diagnostics include:

- the effective rank of the safe logit-difference Jacobian;
- its singular-value spectrum and condition number;
- cosine similarities between constraint gradients from different states; and
- the amount by which changing a decision at one state changes logits at other states.

A low-rank or highly conflicting Jacobian indicates cross-state interference: improving
one safe decision may consume safety margin elsewhere. Such a policy can have a large
minimum margin but still be a poor base for reward improvement.

### 3.8 Optimization conditioning and saturation

Record the following inexpensive neural-network diagnostics:

- fraction of Tanh activations close to `-1` or `+1`;
- parameter norms and layer spectral norms;
- action-logit Jacobian norms and effective rank;
- policy Fisher eigenvalue spectrum or condition number; and
- gradient norms for the safety-separation objective.

Excessive saturation or poor conditioning can slow learning and make interval bounds
loose. These diagnostics are particularly important when an initialization was trained
against large-magnitude logit targets.

### 3.9 Reward-free behavioural coverage, when permitted

If interaction with the environment is allowed before rewards are exposed, the initial
policy can also be assessed using reward-free trajectory statistics:

- state visitation coverage or entropy;
- fraction of shield states visited;
- action diversity within the safe set;
- trajectory length; and
- frequency of entering behavioural dead ends.

These quantities use dynamics but not reward. They should be omitted when the
experimental assumption prohibits any pre-training environment interaction.

## 4. Pushing safe logits high and unsafe logits low

Increasing the safe-unsafe separation is a plausible way to reduce early safety
violations and projection. It may be helpful up to a point. Driving targets to extremely
large positive and negative values is not generally a good solution.

Potential benefits are:

- a larger initial functional safety buffer;
- more verify-first updates accepted without region construction;
- fewer early projections or reverts; and
- reduced risk that the first noisy PPO updates cross a safety boundary.

Potential failure modes are:

- softmax and Tanh saturation;
- vanishing gradients for actions assigned almost zero probability;
- arbitrary commitment to a particular safe action;
- reduced exploration among safe behaviours;
- unnecessarily large weights and poor network conditioning;
- looser interval bounds and a smaller certified parameter region; and
- a large functional margin but a small relative parameter-space radius.

The useful target is therefore not the largest attainable logit margin. It is a
moderate safety buffer achieved by a well-conditioned model that remains flexible among
safe actions.

## 5. A finite soft-safe regression target

For a state with safe action set `S`, unsafe action set `U`, sizes `n_s` and `n_u`, and
a chosen total unsafe probability `epsilon`, define

```text
q(a | s) = (1 - epsilon) / n_s,  if a is safe,
q(a | s) = epsilon / n_u,        if a is unsafe.
```

Fit the initial policy using cross-entropy or `KL(q || pi_theta)`. The finite target
safe-unsafe logit gap is

```text
m_s = log(((1 - epsilon) * n_u) / (epsilon * n_s)).
```

This formulation has several advantages over regressing to arbitrary high and low
logits:

1. `epsilon` has a direct probabilistic interpretation.
2. Safe actions remain neutral in the absence of reward.
3. Target logits remain finite.
4. The required gap automatically accounts for the numbers of safe and unsafe actions.
5. The objective is invariant to a common shift in all logits.

If explicit logit targets are operationally easier, use centred targets whose common
mean is zero. For a requested pairwise gap `m`, one choice is

```text
safe target   = +m * n_u / (n_s + n_u)
unsafe target = -m * n_s / (n_s + n_u).
```

The difference is `m`, but the optimization does not waste capacity learning an
irrelevant common logit offset.

The regression objective can additionally include:

- a safe-logit variance penalty;
- weight decay or spectral regularization;
- a Tanh-saturation penalty; and
- an adversarial parameter-perturbation loss that encourages safety in a relative
  neighbourhood around the base parameters.

## 6. Directly optimizing initialization robustness

Instead of using output margin as a surrogate, the base-policy objective can explicitly
encourage parameter-space robustness. Conceptually:

```text
maximize over theta, r:  r

subject to:
    the policy is safe for every theta' satisfying
    |theta_i' - theta_i| <= r * c_i.
```

Exact joint optimization may be expensive. Practical approximations include:

1. Penalize the inverse linearized relative radius.
2. Adversarially perturb parameters during base-policy fitting.
3. Alternate ordinary soft-safe fitting with certified-radius expansion.
4. Train several bases and select the one with the largest verified initial region.

Safety certification remains the final acceptance test. The robustness objective only
tries to produce a policy for which that test admits a useful neighbourhood.

## 7. Interaction with the PSPO safety semantics

The current all-safe-actions constraint requires every shield-safe action to be ranked
above every unsafe action. This is stronger than requiring only the greedy action to be
safe. It can reduce mobility because a low-ranked but still safe action may become the
bottleneck.

Both quantities should be measured as diagnostics:

- an **all-safe-actions radius**, matching the current conservative certificate; and
- an **any-safe-argmax radius**, measuring the space in which at least one safe action
  remains above every unsafe action.

Changing the certificate is a separate methodological decision. The any-safe condition
is appropriate for a deterministic greedy deployed policy, but it does not guarantee
zero unsafe probability for stochastic action sampling. Diagnostic comparison does not
require weakening the existing PSPO guarantee.

## 8. Candidate selection protocol

For a new environment, a reward-free initialization pipeline could be:

1. Train multiple base policies using several random seeds and moderate soft-safe
   targets, for example `epsilon` in `{1e-1, 1e-2, 1e-3}`.
2. Reject any base that fails exact initial safety.
3. Compute the distribution of functional safety gaps.
4. Compute linearized relative parameter radii.
5. For the most promising candidates, compute a certified radius or one full initial
   PSPO region.
6. Measure conditional safe-action entropy, controllability, Jacobian rank, and Tanh
   saturation.
7. Sample representative safe directions and measure directional mobility.
8. Select candidates on a Pareto frontier rather than maximizing only one metric.
9. Run PSPO from the selected candidates, while logging projection frequency, region
   recomputation frequency, accepted-step size, and final reward.

A possible composite score for exploratory analysis is

```text
score =
    log(p05_relative_safe_radius)
  + lambda_entropy * mean_safe_action_entropy
  + lambda_mobility * log(median_directional_radius)
  - lambda_sat * tanh_saturation_fraction
  - lambda_cond * log(jacobian_condition_number).
```

The score should not initially be treated as universal. Pareto selection is preferable
until enough experiments establish stable scaling and predictive relationships.

Historical rewards may be used after experiments to test which reward-free diagnostics
were predictive. They should not be used to select the initial policy within the same
experiment when preserving the reward-unavailable assumption.

## 9. Evaluation questions

The following relationships should be tested empirically:

1. Does certified relative radius predict the fraction of verify-first updates accepted
   without projection?
2. Does directional mobility predict the retained fraction of PPO update vectors?
3. Does safe-action entropy predict faster reward improvement after training starts?
4. Does a high raw margin achieved through large weights reduce interval-region width?
5. Are projection count and projection distance better explained by parameter-space
   radius than by minimum logit margin?
6. Does safe-action controllability explain differences between initial policies with
   nearly identical margins?
7. Is there a moderate target margin or unsafe-probability target beyond which safety
   improves little but policy plasticity deteriorates?

The most useful early success criterion is not final reward alone. A proposed metric is
valuable if it consistently predicts intermediate PSPO behaviour such as accepted safe
step size, projection distance, number of region computations, and safe-action switching
capacity.

## 10. Recommended implementation order

The lowest-cost, most informative progression is:

1. Extend initial-policy reporting with full gap distributions, safe-action entropy,
   Tanh saturation, and parameter norms.
2. Implement the linearized relative `L-infinity` safety radius and identify its
   bottleneck constraints.
3. Add random and Fisher-normalized directional-radius probes.
4. Train soft-uniform-safe initial policies with finite `epsilon` targets.
5. Compute one initial certified region for shortlisted policies.
6. Add safe-action controllability and Jacobian-spectrum diagnostics.
7. If the approximations correlate with PSPO behaviour, investigate direct robust-radius
   optimization during base-policy fitting.

## 11. Main hypothesis

The principal hypothesis is:

> A moderately confident initial policy that clearly separates unsafe actions, remains
> neutral among safe actions, and has a large normalized parameter-space safety radius
> will support better reward improvement than a maximally separated but saturated
> policy.

This keeps the core PSPO idea intact. Safety still follows from verified policy updates,
while the initialization is chosen to make safe traversal of parameter space easier and
less dependent on projection.
