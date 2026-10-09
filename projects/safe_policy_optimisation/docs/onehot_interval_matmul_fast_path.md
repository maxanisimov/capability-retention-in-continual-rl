# One-hot fast path in the interval matmul

Measured 2026-09-21. Changes `core/abstract_gradient_training/interval_arithmetic.py`;
tests in `core/abstract_gradient_training/test_interval_arithmetic_unittest.py`;
benchmark in `core/scripts/benchmark_interval_matmul_onehot.py`.

## What the certificate was doing

For a tabular environment the certificate runs in `points` mode over one-hot
state encodings, so the actor's first linear layer evaluates

```
(batch x |S|) @ (|S| x hidden)
```

through `propagate_matmul_rump`, which computes

```
H_mu = A_mu @ B_mu
H_r  = |A_mu| @ B_r + A_r @ |B_mu| + A_r @ B_r
```

The left operand is a point (`A_l == A_u`), so `A_r` is exactly zero and two of
those four matmuls multiply a zero matrix. The operand is also one-hot, so the
remaining two are row lookups. Profiling at 256x256 had already put 83.7% of
runtime in this one function.

## Two fast paths

1. **Point operand** (`propagate_matmul_rump`). If either operand is degenerate,
   the terms carrying its radius vanish: two matmuls instead of four. This is
   **bit-identical** for finite inputs, since `(a + a) / 2 == a` and
   `(a - a) / 2 == 0` exactly.
2. **One-hot rows** (`propagate_matmul`, before method dispatch). If `A` is a
   point matrix of one-hot rows, `A @ [B_l, B_u]` is exactly
   `[B_l[idx], B_u[idx]]`, so the whole matmul becomes an `index_select`,
   constant in `|S|`. It is exact for all three methods (`rump`, `exact`,
   `nguyen`), so it is applied ahead of the dispatch.

The lookup is **not** bit-identical to the old path: the old one reached the
result through a midpoint/radius round trip, and `(l + u) / 2 -/+ (u - l) / 2`
does not reproduce `l` and `u`. The lookup returns the exact image, so bounds
move by at most one ulp and can only get tighter. In the end-to-end runs below,
the projection displacement was unchanged to all 16 digits.

**Detection is exact, not tolerance-based.** A row-sum test is ~50x faster but a
row of `[1.0, 1e-20, 0, ...]` sums to exactly `1.0` in float32, which would
silently drop that entry's contribution to the bound. `nonzero()` enumerates the
entries instead — 5x faster than `count_nonzero(dim=1)` and not foolable.

## Per-call cost, single-threaded, float32

`baseline` is Rump's algorithm as it stood before this change; `point` is fast
path 1 alone; `fast` is both. Shapes are the real certificate shapes (`hidden`
= 64, batch = the number of winning states).

| shape | baseline | point | fast | speedup | peak RSS base -> fast |
| --- | --- | --- | --- | --- | --- |
| 1024 x 16384 | 0.397 s | 0.202 s | 0.00045 s | 874x | 0.86 -> 0.69 GiB |
| 1024 x 65536 | 1.602 s | 0.794 s | 0.0019 s | 843x | 1.88 -> 1.26 GiB |
| **14080 x 16384** (FL128) | 5.61 s | 2.76 s | 0.00077 s | **7300x** | 4.84 -> 3.08 GiB |
| **56320 x 65536** (FL256) | 110.8 s | 54.1 s | 0.0058 s | **19000x** | 69.4 -> 41.8 GiB |

The residual RSS is the one-hot input tensor itself (14.7 GB at FL256); what the
fast path removes is the three `batch x |S|` temporaries built on top of it.

## End to end

One real region computation (`--smoke`, `n_iters=2`, one pinned core), measured
by swapping the fast paths out in-process.

FrozenLake **128** (14,080 certificate states):

| variant | LID time | vs baseline |
| --- | --- | --- |
| baseline (pre-change) | 124.57 s | 1.00x |
| point operand only | 77.98 s | 1.60x |
| both fast paths | **48.91 s** | **2.55x** |

FrozenLake **256** (56,320 certificate states):

| variant | LID time | peak RSS | vs baseline |
| --- | --- | --- | --- |
| baseline (pre-change) | 2236.88 s | 103.4 GB | 1.00x |
| both fast paths | **866.73 s** | **74.5 GB** | **2.58x** |

`projection_displacement_l2` was identical to all 16 digits within each scale
(`0.3322225164248666` at 128, `0.3607451527148167` at 256), so the enforcement
decision is unchanged. The 103.4 GB baseline peak corroborates the 98.7 GiB per
seed measured during the production 256 sweep.

**What this is worth on the 256 sweep.** The per-seed budget was 15.5 h of PPO
plus 9 enforcement events at ~52 min = ~23 h. At 2.58x the events cost ~3.0 h
instead of ~7.8 h, so **~18.5 h/seed** (projected from these measurements, not
yet observed on a full seed). The memory saving matters more: at ~71 GiB rather
than ~98.7 GiB per worker, the concurrency cap on a 1007 GiB machine rises from
6 seeds to 8-9.

## What still costs, and why the speedup is 2.55x rather than 7300x

Detection accounts for 6.35 s of the remaining 48.91 s at 128 scale (102.9 s of
866.73 s at 256), because the certificate
**allocates a fresh one-hot batch on every call** — 10 distinct tensor objects
across 13 calls, with pointers recycled in between. A memoisation keyed on
operand identity and version counter was implemented and measured: it never hit,
and was removed rather than left in the certification path as untriggered
complexity.

The rest is the other layers, whose left operand is a genuine interval, plus the
Rashomon loop's own work.

**The remaining prize is upstream.** Materialising a dense `batch x |S|` one-hot
matrix per call costs O(|S|) memory per certificate row regardless of how fast
the matmul is, and that — not the matmul — is what makes 512 and 1024 impossible
(the one-hot certificate alone is 220 GiB and 3.44 TiB). Passing state *indices*
to the bounded model and letting the first layer gather would remove both the
allocation and the detection scan. This fast path is the enabler for that: it
establishes the lookup is exact and isolates where the gather belongs.

That upstream path is now available as
`--state-representation state_id_lookup`. The certificate stores an `(N, 1)`
integer tensor and `StateIdLookupLinear` gathers the matching lower and upper
first-layer weight columns. The dense `one_hot` mode remains available for
equivalence checks and historical reproduction.

See `pspo-frozenlake-scalability` for the measured scaling costs.
