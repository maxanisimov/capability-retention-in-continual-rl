"""Tests for the segment safe-region search (the longest certified safe step)."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from provably_safe_policy_optimisation.regions import (
    ZonotopeRegion,
    flatten_tensors,
    unflatten_like,
)
from src.segment_rashomon import (
    certify_segment,
    compute_segment_rashomon_set,
    segment_endpoint,
    select_certified_segment,
)


def _model(seed: int = 0, hidden: int = 8, n_in: int = 6, n_out: int = 3):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, n_out),
    )


def _flat(model) -> torch.Tensor:
    return flatten_tensors([p.detach() for p in model.parameters()])


def _set_flat(model, flat: torch.Tensor) -> None:
    params = list(model.parameters())
    for param, value in zip(params, unflatten_like(flat, params)):
        param.data.copy_(value)


def _safe_dataset(model, n: int = 24, n_in: int = 6, n_out: int = 3) -> TensorDataset:
    """States labelled with the model's own greedy action as the only safe one.

    Guarantees theta_k itself certifies, which is the induction hypothesis the
    segment search relies on.
    """
    torch.manual_seed(7)
    inputs = torch.randn(n, n_in)
    with torch.no_grad():
        greedy = model(inputs).argmax(dim=1)
    targets = torch.nn.functional.one_hot(greedy, num_classes=n_out).float()
    return TensorDataset(inputs, targets)


def _greedy_safe(model, flat: torch.Tensor, dataset: TensorDataset) -> bool:
    """Exact greedy check at one parameter vector."""
    inputs, targets = dataset.tensors
    saved = _flat(model).clone()
    try:
        _set_flat(model, flat)
        with torch.no_grad():
            chosen = model(inputs).argmax(dim=1)
        return bool(targets[torch.arange(len(chosen)), chosen].all())
    finally:
        _set_flat(model, saved)


class SegmentRashomonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model()
        self.dataset = _safe_dataset(self.model)
        self.theta_k = _flat(self.model).clone()

    def _delta(self, scale: float, seed: int = 3) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        params = list(self.model.parameters())
        flat = torch.randn(self.theta_k.numel(), generator=generator)
        flat = flat / flat.norm() * scale
        return unflatten_like(flat, params)

    def test_rank_one_region_is_exactly_the_segment(self) -> None:
        """Proposition 1: Z(0, alpha) reproduces theta_k + t delta."""
        delta = self._delta(0.4)
        result = compute_segment_rashomon_set(
            self.model, self.dataset, delta=delta, splits=1
        )
        self.assertTrue(result.regions, "expected a certified segment")
        region = result.regions[0]
        centre = flatten_tensors(region.center_params)
        generator = region.generators.reshape(-1)
        delta_flat = flatten_tensors(delta)
        for z in torch.linspace(-1.0, 1.0, 11):
            t = result.alpha * 0.5 * (1.0 + float(z))
            torch.testing.assert_close(
                centre + z * generator, self.theta_k + t * delta_flat
            )

    def test_certified_step_is_sound(self) -> None:
        """The load-bearing property: the whole path to the endpoint is safe."""
        for scale in (0.2, 0.8, 2.0):
            with self.subTest(scale=scale):
                delta = self._delta(scale)
                result = compute_segment_rashomon_set(
                    self.model, self.dataset, delta=delta, splits=2
                )
                self.assertTrue(result.regions, "expected a certified segment")
                self.assertGreater(result.alpha, 0.0)
                delta_flat = flatten_tensors(delta)
                for t in torch.linspace(0.0, result.alpha, 41):
                    self.assertTrue(
                        _greedy_safe(
                            self.model, self.theta_k + float(t) * delta_flat, self.dataset
                        ),
                        f"unsafe policy at t={float(t):.4f} inside a certified segment",
                    )

    def test_certified_step_never_exceeds_the_true_maximum(self) -> None:
        """alpha_hat <= alpha_star, measured by dense sampling."""
        delta = self._delta(3.0)
        delta_flat = flatten_tensors(delta)
        alpha_star = 1.0
        for t in torch.linspace(0.0, 1.0, 2001):
            if not _greedy_safe(self.model, self.theta_k + float(t) * delta_flat, self.dataset):
                alpha_star = float(t)
                break
        result = compute_segment_rashomon_set(
            self.model, self.dataset, delta=delta, splits=4
        )
        self.assertLessEqual(result.alpha, alpha_star + 1e-9)

    def test_certificate_is_monotone_in_alpha(self) -> None:
        """Proposition 3: C_K never goes 0 -> 1 as alpha shrinks."""
        delta = self._delta(3.0)
        delta_flat = flatten_tensors(delta)
        batches = [tuple(self.dataset.tensors)]
        # Certified at alpha implies certified at every smaller alpha. The scale
        # is chosen so both outcomes occur across the sweep.
        outcomes = [
            (
                float(alpha),
                certify_segment(
                    self.model, self.theta_k, delta_flat, float(alpha), 1, batches
                ),
            )
            for alpha in torch.linspace(0.0, 1.0, 41)
        ]
        self.assertTrue(any(certified for _, certified in outcomes))
        self.assertFalse(all(certified for _, certified in outcomes))
        for index, (alpha, certified) in enumerate(outcomes):
            if certified:
                for smaller, smaller_certified in outcomes[:index]:
                    self.assertTrue(
                        smaller_certified,
                        f"C_K({smaller:.4f}) failed but C_K({alpha:.4f}) certified",
                    )

    def test_more_splits_never_shorten_the_step(self) -> None:
        """Proposition 5: refinement is sound and monotone."""
        delta = self._delta(3.0)
        previous = -1.0
        for splits in (1, 2, 4, 8):
            result = compute_segment_rashomon_set(
                self.model,
                self.dataset,
                delta=delta,
                splits=splits,
                max_splits=splits,
                tolerance=1e-4,
            )
            self.assertGreaterEqual(result.alpha, previous - 1e-6)
            previous = result.alpha

    def test_safe_proposal_takes_the_fast_path(self) -> None:
        """A tiny step certifies at alpha = 1 in a single evaluation."""
        result = compute_segment_rashomon_set(
            self.model, self.dataset, delta=self._delta(1e-5), splits=1
        )
        self.assertEqual(result.alpha, 1.0)
        self.assertEqual(result.iterations_run, 1)
        self.assertTrue(result.target_contained_and_certified)

    def test_hopeless_direction_fails_closed(self) -> None:
        """No positive certified step returns no region rather than a guess."""
        # Drive the logits hard towards a deliberately unsafe action.
        params = list(self.model.parameters())
        delta_flat = torch.zeros(self.theta_k.numel())
        delta_flat[-params[-1].numel():] = 1e4
        delta = unflatten_like(delta_flat, params)
        result = compute_segment_rashomon_set(
            self.model, self.dataset, delta=delta, splits=1, tolerance=1e-2
        )
        self.assertEqual(result.alpha, 0.0)
        self.assertEqual(result.regions, [])
        self.assertIsNone(select_certified_segment(result))

    def test_model_parameters_are_restored(self) -> None:
        """The search mutates the model as scratch space but must hand it back."""
        compute_segment_rashomon_set(
            self.model, self.dataset, delta=self._delta(3.0), splits=2
        )
        torch.testing.assert_close(_flat(self.model), self.theta_k)

    def test_endpoint_matches_the_certified_step(self) -> None:
        delta = self._delta(0.5)
        result = compute_segment_rashomon_set(self.model, self.dataset, delta=delta)
        region = result.regions[0]
        endpoint = flatten_tensors(segment_endpoint(region))
        torch.testing.assert_close(
            endpoint, self.theta_k + result.alpha * flatten_tensors(delta)
        )

    def test_select_certified_segment_returns_the_region(self) -> None:
        result = compute_segment_rashomon_set(
            self.model, self.dataset, delta=self._delta(0.3)
        )
        selected = select_certified_segment(result)
        self.assertIsNotNone(selected)
        region, index = selected  # type: ignore[misc]
        self.assertIsInstance(region, ZonotopeRegion)
        self.assertEqual(index, 0)
        self.assertEqual(int(region.generators.shape[0]), 1)

    def _interval_dataset(self, eps: float) -> TensorDataset:
        inputs, targets = self.dataset.tensors
        return TensorDataset(inputs - eps, inputs + eps, targets)

    def test_interval_inputs_are_certified_soundly(self) -> None:
        """Every (theta, x) in the certified segment x input box must be safe.

        Continuous shields certify over input boxes, so soundness has to hold
        jointly in the parameter and the input coordinate, not just at the box
        centres the point-input tests use.
        """
        eps = 1e-3
        delta = self._delta(0.2)
        result = compute_segment_rashomon_set(
            self.model,
            self._interval_dataset(eps),
            delta=delta,
            has_input_intervals=True,
        )
        self.assertGreater(result.alpha, 0.0, "expected a non-vacuous certificate")

        inputs, targets = self.dataset.tensors
        delta_flat = flatten_tensors(delta)
        saved = _flat(self.model).clone()
        generator = torch.Generator().manual_seed(11)
        try:
            for t in torch.linspace(0.0, result.alpha, 9):
                _set_flat(self.model, self.theta_k + float(t) * delta_flat)
                for _ in range(8):
                    offset = (
                        torch.rand(inputs.shape, generator=generator) * 2.0 - 1.0
                    ) * eps
                    with torch.no_grad():
                        chosen = self.model(inputs + offset).argmax(dim=1)
                    self.assertTrue(
                        bool(targets[torch.arange(len(chosen)), chosen].all()),
                        f"unsafe action at t={float(t):.4f} inside a certified segment",
                    )
        finally:
            _set_flat(self.model, saved)

    def test_interval_alpha_is_non_increasing_in_box_width(self) -> None:
        """A wider input box can only shrink the certified step.

        This is what makes the interval support non-vacuous: if the upper bound
        were dropped and the box collapsed to its centre, alpha would not move.
        """
        delta = self._delta(0.2)
        alphas = []
        for eps in (0.0, 1e-4, 1e-3, 5e-3):
            result = compute_segment_rashomon_set(
                self.model,
                self._interval_dataset(eps),
                delta=delta,
                has_input_intervals=True,
            )
            alphas.append(result.alpha)
        for narrow, wide in zip(alphas, alphas[1:]):
            self.assertLessEqual(wide, narrow + 1e-12)
        self.assertGreater(
            alphas[0],
            alphas[2],
            "box width did not affect the certificate; x_u may be ignored",
        )

    def test_rejects_mismatched_delta(self) -> None:
        with self.assertRaises(ValueError):
            compute_segment_rashomon_set(
                self.model, self.dataset, delta=[torch.zeros(2)]
            )

    def test_rejects_invalid_search_settings(self) -> None:
        delta = self._delta(0.1)
        with self.assertRaises(ValueError):
            compute_segment_rashomon_set(
                self.model, self.dataset, delta=delta, tolerance=0.0
            )
        with self.assertRaises(ValueError):
            compute_segment_rashomon_set(
                self.model, self.dataset, delta=delta, splits=0
            )
        with self.assertRaises(ValueError):
            compute_segment_rashomon_set(
                self.model, self.dataset, delta=delta, splits=4, max_splits=2
            )


if __name__ == "__main__":
    unittest.main()
