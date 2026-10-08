# When CROWN is looser than IBP on Tanh networks

**Summary.** A tighter verification method is not uniformly tighter. For our
Tanh actors, CROWN beats IBP when the weights are fixed and only the *input* is
an interval, but the ordering **inverts** once the weights are also intervals
and those intervals get wide — which is exactly the regime PSPO's certified
region growth operates in. On the LunarLander descent certificate the crossover
sits at a parameter radius of roughly `3e-3`. Above it, CROWN's bounds degrade
about twice as fast as IBP's, so the largest box CROWN can certify is *smaller*
than IBP's, and PSPO ends up rejecting most of its updates.

This note explains the three separate effects behind that, each measured.

## Where this came from

The 2026-09-09 LunarLander descent-shield runs
(`outputs/continuous_state_shields/pspo/20260909_lunarlander_descent_{loose,crown}/`)
were identical apart from the verifier. Switching IBP → CROWN made things worse
on every axis that matters:

| per seed, 245 PPO updates | IBP | CROWN |
|---|---|---|
| Certified regions computed | 245 | 65 |
| Projections applied | 245 | 62 |
| **Projection failures** | **0** | **183** |
| Typical certified region size | 159 → 232 (growing) | 95 → 107 (flat) |
| Mean reward | −175.4 | −342.8 |
| Wall time | 1112 s | ~1560 s |

CROWN paid ~1.4× the compute for roughly half the certified region. The runs
also emit, once, the warning that names the first of the three effects:

```
Tanh CROWN bounds encountered mixed-sign intervals. Using a sound
single-affine residual fallback; bounds may be looser than split-domain bounds.
```

## Background: what the two methods actually compute

**IBP** (`IntervalBoundedModel`) propagates one interval per neuron. At a Tanh
it maps `[l, u] -> [tanh(l), tanh(u)]`.

**CROWN** (`CROWNBoundedModel`) keeps a *linear* function of the network input
and only concretises at the end. At each Tanh it must first replace the
non-linearity by an affine envelope

```
a_l · z + b_l  ≤  tanh(z)  ≤  a_u · z + b_u        for all z in [l, u]
```

and then propagates those coefficients backwards. Its advantage is that
correlated contributions from different neurons can cancel before concretisation
— IBP throws that dependency information away at every layer.

The trade is therefore explicit: **CROWN takes on per-neuron relaxation slack in
exchange for cross-layer dependency cancellation.** It is tighter only when the
cancellation is worth more than the slack.

## Effect 1: Tanh is the worst case for a linear relaxation

`tanh` is **monotone**, so IBP's image `[tanh(l), tanh(u)]` is not an
over-approximation at all — it is the *exact* range of that neuron given its
input interval. There is no slack for CROWN to recover. Any affine envelope,
concretised on that neuron alone, is strictly wider.

This is the structural reason Tanh differs from, say, ReLU in a deep network
with many unstable neurons, where interval propagation compounds badly and CROWN
has real slack to win back.

Measured with the repo's own `tanh_linear_bounds`, concretising the envelope
over the same input interval:

| input interval | IBP width | CROWN width | ratio | secant slope | envelope gap `b_u − b_l` |
|---|---|---|---|---|---|
| [−0.25, 0.25] | 0.4898 | 0.4937 | 1.01× | 0.980 | 0.004 |
| [−1, 1] | 1.5232 | 1.6867 | 1.11× | 0.762 | 0.164 |
| [−2, 2] | 1.9281 | 2.4931 | 1.29× | 0.482 | 0.565 |
| [−5, 5] | 1.9998 | 3.2113 | **1.61×** | 0.200 | 1.212 |
| [−20, 20] | 2.0000 | 3.7315 | **1.87×** | 0.050 | 1.732 |
| [0.5, 5] (one-sided) | 0.5378 | 0.8677 | 1.61× | — | — |
| [1, 3] (one-sided) | 0.2335 | 0.3257 | 1.40× | — | — |

Two things to read off:

- The slack grows with interval width. IBP saturates at 2 (the range of `tanh`);
  the envelope keeps widening because its slope term contributes
  `|a| · (u − l)` on top of the constant gap.
- It is **not only** the mixed-sign case. The one-sided rows use the good
  secant/tangent envelope on a genuinely convex or concave piece and are still
  40–60 % wider than exact.

> Caveat: CROWN does *not* concretise per neuron in practice, so this table is
> not a like-for-like comparison of the two verifiers. It quantifies the slack
> CROWN takes on, which its dependency tracking then has to earn back.

## Effect 2: the mixed-sign fallback

`tanh` has an inflection at 0, so on an interval with `l < 0 < u` it is neither
convex nor concave and no secant/tangent construction applies. Tight verifiers
normally handle this by **splitting the neuron into sub-domains** and bounding
each piece. This CROWN graph cannot — `tanh_linear_bounds` says so directly:

> Mixed intervals use a sound global residual envelope because this CROWN graph
> does not support splitting one neuron into multiple domains.

The fallback (`_residual_bounds` in
`core/abstract_gradient_training/bounded_models/_crown_bounds/tanh_node.py`) uses
**two parallel lines**: both bounds take the secant slope
`a = (tanh(u) − tanh(l)) / (u − l)`, and the offsets are the exact extrema of the
residual `r(z) = tanh(z) − a·z` over `[l, u]`. Sound, but the vertical gap is a
constant that approaches the full output range of `tanh` as the interval widens
(1.73 of a possible 2.0 at [−20, 20] above).

For our LunarLander certificate the input box spans 5 to 20 units per coordinate,
so first-layer pre-activations are massively mixed-sign and essentially every
Tanh neuron takes this path.

**`alpha-CROWN` does not rescue it.** The optimisable relaxation only tunes the
*slope*; the offset is still pinned to the exact residual extremum
(`TanhNode.update_relaxation`). When the residual range is already near the
function's full range there is little left to optimise. Measured on this
certificate, alpha-CROWN reproduced CROWN's result exactly while costing ~200×
per refinement (216 s vs 1 s), which rules it out inside a training loop.

## Effect 3: the inversion, and why it is the one that bit us

Effects 1 and 2 are costs, not a verdict — and with **fixed weights** CROWN's
dependency tracking does pay for them. On the real descent certificate and the
real base policy, bounding the logits over the input box alone:

| boxes | method | min margin | mean logit width |
|---|---|---|---|
| full band (1 box) | IBP | 2.0355 | 1.28 |
| full band (1 box) | **CROWN** | **2.5616** | **1.13** |
| band split y,v_y 8×8 | IBP | 2.0753 | 0.29 |
| band split y,v_y 8×8 | **CROWN** | **2.8273** | **0.13** |

CROWN is clearly tighter. So on its own this says CROWN should have *helped*.

The difference is that PSPO does not certify over an input box. It certifies
over an input box **and a parameter box** — the Rashomon region it is growing.
Repeating the measurement with parameter intervals of increasing radius:

| parameter radius | IBP margin | CROWN margin | tighter |
|---|---|---|---|
| 0 | 2.0355 | 2.5616 | CROWN |
| 1e-4 | 1.9895 | 2.5096 | CROWN |
| 1e-3 | 1.5524 | 1.9907 | CROWN |
| 3e-3 | 0.4433 | 0.5362 | CROWN |
| **1e-2** | **−3.8886** | **−5.4200** | **IBP** |
| 3e-2 | −8.1158 | −13.5796 | IBP |
| 1e-1 | −17.2411 | −32.3346 | IBP |

There is a **crossover near a parameter radius of 3e-3**. Below it CROWN wins;
above it CROWN degrades roughly twice as fast.

The mechanism is in `TanhNode._backpropagate`. With interval weights the backward
coefficient matrices `Lambda` / `Omega` become *intervals* rather than numbers,
and two things follow:

1. **Sign-dependent envelope selection breaks down.** The node picks the upper
   or lower envelope according to the sign of the propagated coefficient. When a
   coefficient interval straddles zero neither choice is valid, and the code
   falls back to concretising that term outright
   (`(Lambda * mixed_mask) @ conc`). Wider parameter boxes mean more
   straddling coefficients, so more of the linear form collapses to intervals —
   CROWN progressively degenerates toward IBP *while still paying the relaxation
   slack from Effects 1 and 2*.
2. **Interval matrix products over-approximate.** The backward pass multiplies
   interval-valued coefficient matrices layer by layer (Rump's method by
   default). That error compounds multiplicatively with depth, and it applies to
   CROWN's coefficient matrices but not to IBP's simpler per-layer intervals.

**Why PSPO lands on the wrong side of the crossover.** Region growth is a
primal–dual loop that expands the parameter box until the certificate is about
to fail. It therefore deliberately operates at the largest box the verifier will
accept — the far right of that table, never the left. The verifier that
degrades more gracefully with parameter radius wins, and that is IBP. This is
what produced CROWN's smaller, flat regions (95–107 vs IBP's 159→232) and the
183/245 projection failures.

## Practical guidance

- **Default to IBP for region growth over parameter boxes.** For this network
  class it is not merely cheaper, it is tighter in the regime that matters.
- **A tighter verifier is worth trying only where the parameter box is small** —
  a final certificate check at a fixed parameter vector, or verification of a
  trained policy, where the fixed-weight table above applies and CROWN gives a
  ~25 % better margin.
- **Do not read the certified fraction across verifiers.** `_interval_certified_fraction`
  in `core/provably_safe_policy_optimisation/adaptive_safe_ppo.py` now uses the
  configured `rashomon_certification_method`, so a reported "100 % certified"
  means 100 % *under that verifier*. Two runs with different verifiers are not
  directly comparable on that number.
- **If a Tanh network must be verified over wide inputs**, narrow the input box
  rather than upgrading the verifier. Effect 1 is driven by interval width and
  no relaxation available here escapes it; splitting the band helped far more
  (mean logit width 1.28 → 0.29 under IBP) than changing method did.
- **`alpha-CROWN` is not a training-loop option** at ~200× IBP's cost per
  refinement, independent of tightness.

## Reproducing the measurements

The three scripts behind the tables are small and self-contained; they read the
committed certificate builder and a base policy from a completed run:

```python
# Effect 1: single-neuron widths
from abstract_gradient_training.bounded_models._crown_bounds.tanh_node import tanh_linear_bounds
a_l, b_l, a_u, b_u = tanh_linear_bounds(torch.tensor([[-5.0]]), torch.tensor([[5.0]]))

# Effects 2-3: end-to-end margins, with and without parameter intervals
from src.verification.api import build_bounded_model
bounded = build_bounded_model(seq, "CROWN", param_l=lower, param_u=upper)
lb, ub = bounded.bound_forward(x_l, x_u)
```

Run with `PYTHONPATH=core`. The descent box comes from
`projects/safe_policy_optimisation/scripts/build_lunarlander_descent_certificate.py`
(`descent_band_box` / `split_box`), and available verifier names come from
`core/src/verification/registry.py`.

## Related

- `projects/safe_policy_optimisation/docs/methodology/pspo_main.tex` — where the
  hard IBP certificate enters the PSPO update rule.
- `core/abstract_gradient_training/bounded_models/_crown_bounds/tanh_node.py` —
  the relaxation and the mixed-sign fallback.
- `core/src/interval_utils.py::compute_rashomon_set` — `growth_method` and
  `certification_method`, exposed on the continuous stage as `--growth-method`
  and `--certification-method`.
