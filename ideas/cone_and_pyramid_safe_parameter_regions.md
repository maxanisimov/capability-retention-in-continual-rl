# Cone- and Pyramid-Shaped Safe Parameter Regions for PSPO

## Status and scope

This document describes options for constructing a finite safe parameter region whose
tip is the current policy and whose axis follows a candidate policy update. The base can
be either:

- a Euclidean ball, producing a circular cone in a selected parameter subspace; or
- an orthotope, producing a pyramid or polyhedral cone.

The goal is to certify every policy in the region under the same safety specification
used by PSPO. Sampling points from a cone can be useful experimentally, but it is not a
certificate and is therefore not considered a safe-region construction by itself.

## 1. Geometry

Let:

- `theta_0` be the current certified-safe parameter vector;
- `delta_raw` be the candidate update produced by the optimizer;
- `u` be the chosen unit update direction;
- `h` be the certified cone height;
- `B` be a `p x k` lateral basis, with columns orthogonal to `u`;
- `k` be the base dimension, normally much smaller than the parameter count `p`.

The base centre is

```text
theta_base = theta_0 + h * u.
```

If the unmodified candidate is used as the base centre, then
`h = ||delta_raw||` and `u = delta_raw / ||delta_raw||`. In practice, the raw update
should usually define only the direction and an upper bound on the height. Certification
or safe line search should select the final `h`.

### 1.1 Circular base

For an orthonormal lateral basis `B`, a finite circular cone is

```text
C_ball = {
    theta_0 + s * h * u + B * v :
    0 <= s <= 1,
    ||v||_2 <= s * r
}.
```

Here `r` is the base radius. The radius of every cross-section grows linearly from zero
at the tip to `r` at the base.

### 1.2 Rectangular base

Let `a_i` be the base half-width associated with lateral direction `B_i`. The pyramid is

```text
C_box = {
    theta_0 + s * h * u + B * v :
    0 <= s <= 1,
    -s * a_i <= v_i <= s * a_i
}.
```

Both sets are closed and convex. This makes membership testing and Euclidean projection
tractable.

### 1.3 Height and base area are not sufficient

The candidate direction determines the cone axis, but a region also requires:

- the base dimension `k`;
- the orientation of `B`;
- a parameter-space metric;
- the radius for a circular base; or
- every half-width `a_i` for a rectangular base.

For a `k`-dimensional circular base,

```text
base_volume = pi^(k/2) / Gamma(k/2 + 1) * r^k.
```

For a rectangular base,

```text
base_volume = product_i (2 * a_i).
```

The rectangular base volume does not determine its aspect ratio. Two bases with the
same volume can produce very different certificates.

The intrinsic `(k + 1)`-dimensional cone volume is

```text
cone_volume = base_volume * h / (k + 1).
```

Volume is useful as a comparison metric, but it should not be the only optimization
objective because parameter scaling can change it without changing policy behaviour.

## 2. Important safety consequence

The base centre belongs to the cone. Therefore, if the raw candidate policy is the base
centre, certifying the whole cone necessarily certifies that candidate policy:

```text
candidate in cone
and cone is certified safe
implies candidate is certified safe.
```

The cone cannot turn an unsafe candidate into a safe one. Also, projecting the candidate
onto a cone centred on that candidate does nothing because the candidate is already
feasible.

The recommended interpretation is:

1. Use the candidate gradient update to choose the axis.
2. Find a certified height along that axis.
3. Grow the lateral base while preserving the certificate.
4. Use the resulting cone as a trust region for subsequent optimizer steps.

## 3. Choosing the metric and lateral subspace

### 3.1 Parameter-space metric

Raw Euclidean distance treats every parameter equally. This can be inappropriate when
layers have different scales or when the network has parameterization symmetries.
Possible metrics are:

1. **Euclidean metric.** Simple and compatible with the current flattened parameter
   representation, but sensitive to parameter scaling.
2. **Layer-normalized metric.** Scale each layer using its parameter norm or a fixed
   reference scale. This is inexpensive and usually more interpretable.
3. **Optimizer metric.** Use Adam or RMSProp second-moment estimates to whiten parameter
   directions. This aligns the geometry with the update calculation.
4. **Fisher metric.** Use a diagonal or low-rank Fisher approximation. This is more
   closely related to policy behaviour but costs more to estimate and stabilize.

The first implementation should support Euclidean geometry. A metric should be an
explicit extension rather than being hidden inside the cone construction.

### 3.2 Lateral basis options

A full base orthogonal to the update has dimension `p - 1`, which is impractical for a
large network. Use a low-rank basis with configurable rank `k`.

Possible sources for `B` are:

1. Existing certified zonotope generators, orthogonalized against the cone axis.
2. Principal components of recent safe policy updates.
3. Leading eigendirections of a Fisher, gradient-covariance, or Hessian approximation.
4. Recent optimizer directions after layer normalization.
5. Random orthogonal directions as a baseline.

The basis affects the size and usefulness of the region, but it must not be trusted as a
safety argument. Safety comes from certifying the entire resulting region. Stability of
the estimated correlations is therefore an efficiency concern rather than a premise of
the guarantee.

If an estimated basis is reused, useful stability diagnostics include principal angles
between consecutive subspaces, generator norm drift, and certificate retention on a
held-out certification set.

## 4. Region parameterization for search

Use opening slopes rather than searching independently over height and base area:

```text
r = h * rho                         # circular base
a_i = h * rho_i                    # rectangular base
```

For the circular cone, `atan(rho)` is its half-angle. With a fixed axis and basis,
reducing `h`, `rho`, or any `rho_i` gives a nested smaller region. This monotonicity makes
bisection and coordinate-wise growth possible.

A robust search order is:

1. Set the base width to zero and certify the largest safe height on the update ray.
2. Fix that height, initialize all opening slopes to zero, and grow the base.
3. If no nonzero base can be certified, reduce the height and retry.
4. Optimize a size objective only after a feasible cone has been found.

Possible size objectives include log base volume, log cone volume, or a weighted sum of
height and lateral widths. Log-volume avoids one large direction dominating the search.

## 5. Certification options

### Option A: One enclosing zonotope

For a rectangular base, the complete pyramid is contained in the zonotope

```text
Z = {
    theta_0 + 0.5 * h * u
    + xi_0 * 0.5 * h * u
    + sum_i xi_i * a_i * B_i :
    xi_i in [-1, 1]
}.
```

This is an axial prism. It allows full lateral displacement even near the tip, so it is
larger than the pyramid. If this enclosing zonotope is certified safe, the pyramid is
safe as well.

For a circular base, the ball can first be enclosed by a box in the `B` coordinates and
then handled in the same way. This adds another source of conservatism.

Advantages:

- reuses the existing zonotope verifier;
- small implementation change;
- straightforward soundness argument.

Disadvantages:

- usually loose near the tip;
- circular bases become especially conservative as `k` increases;
- rejection of the outer zonotope does not imply that the cone is unsafe.

This is useful as a baseline, but it is unlikely to be the best production method.

### Option B: Slabbed zonotope outer approximation

Partition the axial coefficient into slabs:

```text
0 = s_0 < s_1 < ... < s_m = 1.
```

For slab `[s_l, s_u]`, use an enclosing zonotope with:

```text
axial_center    = theta_0 + 0.5 * (s_l + s_u) * h * u
axial_generator = 0.5 * (s_u - s_l) * h * u
lateral_width_i = s_u * a_i
```

The union of these zonotopes contains the entire pyramid. Certifying every slab therefore
certifies the pyramid. Smaller slabs reduce the false lateral freedom introduced near
the tip.

An adaptive algorithm can split only failed slabs. A failed outer approximation may be
too loose even when the corresponding part of the cone is safe, so refinement should be
attempted before shrinking the cone.

Advantages:

- reuses the current verifier and region-generator representation;
- sound and substantially tighter than one prism;
- tunable cost through the slab count and adaptive refinement;
- naturally parallelizable across slabs.

Disadvantages:

- certification cost grows with the number of slabs;
- still an outer approximation;
- a circular base still requires a polytopic or box enclosure.

This is the recommended first implementation.

### Option C: Direct constrained-zonotope verification

The rectangular pyramid has an affine parameter representation with linear coefficient
constraints:

```text
theta = theta_0 + s * h * u + B * v
0 <= s <= 1
-s * a_i <= v_i <= s * a_i.
```

It can therefore be represented as a constrained zonotope or, more generally, an affine
form over a polytope. Bounds on affine expressions can be calculated with linear
programming.

The existing verifier assumes independently interval-bounded generator coefficients.
Supporting the pyramid directly would require a coefficient-domain abstraction that
preserves the coupling between `s` and `v`, as well as sound handling of nonlinear layer
relaxations and parameter products.

Advantages:

- can represent a rectangular pyramid exactly at the parameter-domain level;
- potentially much tighter than slab enclosures.

Disadvantages:

- significant verifier work;
- repeated LP solves may be expensive;
- nonlinear propagation can still introduce relaxation error.

This option is justified if slab refinement remains the dominant source of failed
certificates or runtime.

### Option D: Direct second-order-cone verification

The circular cone has the coefficient constraint

```text
||v||_2 <= s * r.
```

This is a second-order-cone constraint. For an affine direction `q`, its support bounds
have a compact form:

```text
upper(q) = q^T theta_0
           + max(0, h * q^T u + r * ||B^T q||_2)

lower(q) = q^T theta_0
           + min(0, h * q^T u - r * ||B^T q||_2).
```

These expressions make exact affine concretization possible. A direct implementation
would need a convex-domain or support-function verifier that propagates sound nonlinear
relaxations through the network.

Advantages:

- preserves circular geometry without a box enclosure;
- exact support bounds for affine expressions;
- invariant to rotations within the lateral subspace.

Disadvantages:

- largest verifier change among the options;
- nonlinear propagation and error-term management require careful design;
- likely unnecessary until the slabbed rectangular approach has been evaluated.

### Option E: Sampling or boundary testing

Sampling the interior or base, testing only the candidate, or testing only the cone
boundary cannot certify the complete region for a nonlinear network. Unsafe pockets may
exist between tested points or inside the cone.

Sampling is useful for:

- rejecting clearly bad candidates before expensive certification;
- selecting promising basis directions;
- initializing height and width searches;
- measuring how conservative a sound certificate is.

It must remain a heuristic pre-filter, not the final safety decision.

## 6. Projection options

Projection is easier than certification because both proposed regions are convex.

### 6.1 Rectangular pyramid projection

Given unconstrained parameters `theta_trial`, solve

```text
minimize    ||theta_trial - (theta_0 + s * h * u + B * v)||_2^2
subject to  0 <= s <= 1
            -s * a_i <= v_i <= s * a_i.
```

This is a convex quadratic program with only `k + 1` optimization variables when the
parameter-space residual is reduced using the low-rank basis.

### 6.2 Circular cone projection

Replace the box constraints with

```text
||v||_2 <= s * r.
```

This is a second-order-cone problem. For an orthonormal Euclidean basis, a specialized
closed-form or low-dimensional solver can be used. A general metric requires a small
convex solver or a carefully validated iterative projection.

Projection must include:

- a post-projection feasibility check;
- explicit numerical tolerances;
- a safe fallback to the tip `theta_0` if the solve fails;
- diagnostics for projection distance and active constraints.

## 7. Recommended implementation plan

### Phase 1: Geometry and projection

1. Add a `ConeRegion` representation containing:
   - tip parameters;
   - normalized axis and height;
   - lateral generators;
   - base shape (`box` initially, `ball` later);
   - base half-widths or radius;
   - parameter shapes, order, dtype, and metric metadata.
2. Add strict validation for zero-height updates, non-finite values, basis dimensions,
   orthogonality tolerance, and nonnegative widths.
3. Implement membership checking and rectangular-pyramid projection.
4. Add unit tests in small dimensions where membership and projection can be checked
   analytically.

### Phase 2: Sound certification through slabs

1. Convert a rectangular pyramid into enclosing zonotope slabs.
2. Certify every slab using the existing zonotope forward-bound implementation.
3. Adaptively split failed slabs up to configured depth and compute budget.
4. Accept the cone only when every covering slab has a valid certificate.
5. Store the cone, slab cover, certificate metadata, and verifier configuration together.

### Phase 3: Cone-size search

1. Use the candidate update as the axis and maximum proposed height.
2. Certify the update ray and find a safe height with bisection.
3. Select or estimate a low-rank lateral basis.
4. Grow opening slopes while preserving full slab certification.
5. Optionally alternate between height and width optimization.
6. Report both geometric size and certificate tightness.

### Phase 4: PSPO integration

1. Add `cone` as a safe-region shape without changing the default.
2. Add separate options for base shape, lateral rank, slab count, refinement depth,
   metric, and size-search budget.
3. Attach the cone to the projected optimizer only after successful certification.
4. Recompute the cone after the configured number of optimizer steps or after leaving
   the current trust-region schedule.
5. Preserve the last valid safe region if recomputation fails.

### Phase 5: Evaluate tighter domains only if needed

Measure:

- certificate success rate;
- certified height and base volume;
- number of slab refinements;
- certification wall time;
- projection frequency and distance;
- achieved return under the same safety specification.

Implement direct constrained-zonotope or second-order-cone verification only if these
measurements show that outer-approximation looseness, rather than the underlying policy
safety margin, is the limiting factor.

## 8. Suggested initial configuration

The first experiment should use:

```text
region shape:             cone
base shape:               box
lateral rank:             small, for example 4 to 16
metric:                   Euclidean
certification:            adaptive zonotope slabs
initial slab count:       4
maximum refinement depth: 3 to 5
height search:            bisection on the candidate direction
width search:             shared opening slope, followed by per-direction refinement
failure fallback:         last certified region or the current safe policy
```

A shared opening slope makes the initial search stable and low-dimensional. Individual
half-widths can be optimized after the approach is shown to certify useful regions.

## 9. Safety invariants

Any implementation should enforce the following invariants:

1. The cone tip is a policy already certified under the active safety specification.
2. The cone is installed only after a sound outer cover or direct cone certificate has
   passed for every required group and state.
3. The optimizer projects after every parameter update while the cone is active.
4. Projection failure returns to a known safe point rather than accepting an unchecked
   update.
5. Certificates are invalidated if parameter ordering, architecture, safety targets,
   shield data, or verifier settings change.
6. Reusing a correlation basis never replaces certifying the resulting region.
7. The reported guarantee has the same scope as the underlying PSPO certificate. It does
   not establish safety outside the states, groups, and admissible-action specification
   covered by that certificate.

## 10. Recommendation

Start with a low-rank rectangular pyramid and certify it through adaptive zonotope slab
enclosures. This approach provides:

- a sound certificate using the current verification machinery;
- correlated, update-aligned parameter movement;
- tractable projection;
- a direct path to measuring whether a specialized cone verifier would be worthwhile.

The circular cone should initially remain an experimental follow-up. Its geometry is
appealing, but preserving that geometry during certification requires either a
conservative box enclosure or a new second-order-cone-aware verifier.
