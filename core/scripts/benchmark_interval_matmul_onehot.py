"""Benchmark the one-hot fast path in `interval_arithmetic.propagate_matmul`.

The safety certificate for a tabular environment feeds a batch of one-hot state
encodings through the actor, so the first linear layer computes
`(batch x |S|) @ (|S| x hidden)` with a point one-hot left operand. The general
interval matmul costs O(batch * |S| * hidden) and allocates several
`batch x |S|` temporaries; the fast path is a row lookup, constant in |S|.

Three variants are measured, one per process so the reported peak RSS is
attributable:

    baseline  Rump's algorithm as it stood before the fast paths: four matmuls
              over `batch x |S|` operands plus the radius temporaries.
    point     only the point-operand short-circuit, which drops the two terms
              carrying the (zero) radius of A: two matmuls.
    fast      the one-hot row lookup, constant in |S|.

    python core/scripts/benchmark_interval_matmul_onehot.py --states 65536
    python core/scripts/benchmark_interval_matmul_onehot.py --states 65536 --variant baseline
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from unittest import mock

import torch
from abstract_gradient_training import interval_arithmetic as ia


def baseline_matmul_rump(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rump's algorithm without any operand specialisation, as the reference."""
    A_mu, A_r = (A_u + A_l) / 2, (A_u - A_l) / 2
    B_mu, B_r = (B_u + B_l) / 2, (B_u - B_l) / 2
    H_mu = A_mu @ B_mu
    H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
    return H_mu - H_r, H_mu + H_r


def build_inputs(
    states: int, batch: int, hidden: int, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch of one-hot rows and an interval weight matrix."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(states, (batch,), generator=generator)
    A = torch.nn.functional.one_hot(indices, states).to(dtype)
    B_mu = torch.randn(states, hidden, generator=generator, dtype=dtype)
    B_r = 0.01 * torch.rand(states, hidden, generator=generator, dtype=dtype)
    return A, B_mu - B_r, B_mu + B_r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=16384, help="|S|, the input width")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--variant", choices=["fast", "point", "baseline"], default="fast"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=0, help="0 leaves the torch default"
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    dtype = getattr(torch, args.dtype)
    A, B_l, B_u = build_inputs(
        args.states, args.batch, args.hidden, dtype, args.seed
    )
    A_u = A.clone()

    if args.variant == "baseline":
        run = lambda: baseline_matmul_rump(A, A_u, B_l, B_u)  # noqa: E731
        context = mock.patch.object(ia, "_unused_sentinel", create=True)
    else:
        run = lambda: ia.propagate_matmul(A, A_u, B_l, B_u, "rump")  # noqa: E731
        context = (
            mock.patch.object(ia, "_point_onehot_row_indices", return_value=None)
            if args.variant == "point"
            else mock.patch.object(ia, "_unused_sentinel", create=True)
        )

    with context:
        # The first call pays for detecting the one-hot operand; later calls on
        # the same input reuse it, which is what the Rashomon loop does.
        started = time.perf_counter()
        H_l, H_u = run()
        cold_s = time.perf_counter() - started
        checksum = float(H_l.sum() + H_u.sum())
        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            run()
            timings.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "variant": args.variant,
                "states": args.states,
                "batch": args.batch,
                "hidden": args.hidden,
                "dtype": args.dtype,
                "threads": torch.get_num_threads(),
                "cold_s": cold_s,
                "best_s": min(timings),
                "median_s": sorted(timings)[len(timings) // 2],
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**2,
                "checksum": checksum,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
