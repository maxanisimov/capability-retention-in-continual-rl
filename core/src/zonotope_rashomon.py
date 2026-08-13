"""Zonotope safe-parameter-region search utilities."""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
import tqdm
from torch.utils.data import DataLoader

from provably_safe_policy_optimisation.regions import (
    ZonotopeRegion,
    zonotope_rank_default,
)
from src.IntervalTensor import IntervalTensor
from src.rashomon_spec import RashomonCertificate
from src.verification import verify
from src.verification.verify import bound_forward_pass


@dataclasses.dataclass
class ZonotopeRashomonResult:
    """Result of a zonotope safe-region search."""

    regions: list[ZonotopeRegion]
    certificates: list[list[RashomonCertificate]]
    temperatures: dict[int | None, float]
    surrogate: str = "auto"
    resolved_surrogate: str = "probability"


def _unpack_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) != 2:
        raise ValueError("Zonotope Rashomon datasets must yield (inputs, multi_hot_actions).")
    return batch[0], batch[1]


def _certificate(
    model: torch.nn.Sequential,
    generators: torch.Tensor,
    coefficients: IntervalTensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    tau: float,
    multi_label_mode: Literal["any", "all"],
    surrogate: verify.SurrogateForm,
) -> RashomonCertificate:
    logits = bound_forward_pass(
        model,
        generators,
        coefficients,
        inputs,
        use_zonotopes=True,
    )
    hard = verify.bound_multi_label_accuracy(
        logits,
        targets,
        lower=True,
        aggregation="min",
        mode=multi_label_mode,
    )
    soft = verify.bound_multi_label_accuracy_margin(
        logits,
        targets,
        tau=float(tau),
        lower=True,
        aggregation="min",
        mode=multi_label_mode,
        surrogate=surrogate,
    )
    return RashomonCertificate(
        group=None,
        min_surrogate=float(soft.detach().item()),
        min_hard_acc=float(hard.detach().item()),
    )


def compute_zonotope_rashomon_set(
    model: torch.nn.Sequential,
    dataset: torch.utils.data.Dataset,
    *,
    rank: int | None = None,
    n_iters: int = 2000,
    checkpoint: int = 100,
    batch_size: int = 500,
    certificate_samples: int = 1000,
    inverse_temp: int = 1,
    seed: int = 0,
    multi_label_mode: Literal["any", "all"] = "any",
    surrogate: verify.SurrogateForm = "auto",
    init_scale: float = 1e-4,
    max_generator_abs: float = 10.0,
) -> ZonotopeRashomonResult:
    """Learn a low-rank zonotope around ``model`` and certify checkpoints."""

    if multi_label_mode not in ("any", "all"):
        raise ValueError(f"multi_label_mode must be 'any' or 'all', got {multi_label_mode!r}.")
    resolved_surrogate = verify.resolve_surrogate_form(multi_label_mode, surrogate)
    params = [param.detach() for param in model.parameters()]
    n_params = int(sum(param.numel() for param in params))
    if n_params <= 0:
        raise ValueError("Cannot compute a zonotope for a model without parameters.")
    rank = zonotope_rank_default(n_params) if rank is None else int(rank)
    if rank <= 0:
        raise ValueError(f"Zonotope rank must be positive, got {rank}.")
    rank = min(rank, n_params)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    generators = torch.randn(rank, n_params, generator=generator, dtype=dtype).to(device)
    generators = generators / generators.norm(dim=1, keepdim=True).clamp_min(1e-12)
    generators = (float(init_scale) * generators).requires_grad_(True)
    coeff_l = -torch.ones(rank, device=device, dtype=dtype)
    coeff_u = torch.ones(rank, device=device, dtype=dtype)
    coeffs = IntervalTensor(coeff_l, coeff_u)
    tau = 1.0 / float(inverse_temp)
    temperatures = {None: float(tau)}

    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    cert_loader = DataLoader(
        dataset,
        batch_size=int(certificate_samples),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    cert_inputs, cert_targets = _unpack_batch(next(iter(cert_loader)))
    cert_inputs = cert_inputs.to(device=device, dtype=dtype)
    cert_targets = cert_targets.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam([generators], lr=0.1 / max(1, n_params))
    regions: list[ZonotopeRegion] = []
    certificates: list[list[RashomonCertificate]] = []

    def checkpoint_region() -> None:
        cert = _certificate(
            model,
            generators,
            coeffs,
            cert_inputs,
            cert_targets,
            tau=tau,
            multi_label_mode=multi_label_mode,
            surrogate=surrogate,
        )
        regions.append(
            ZonotopeRegion(
                center_params=[param.detach().cpu().clone() for param in model.parameters()],
                generators=generators.detach().cpu().clone(),
                coefficient_l=coeff_l.detach().cpu().clone(),
                coefficient_u=coeff_u.detach().cpu().clone(),
                param_shapes=[tuple(param.shape) for param in model.parameters()],
            )
        )
        certificates.append([cert])

    data_iter = iter(dataloader)
    checkpoint = int(checkpoint)
    for iter_idx in (pbar := tqdm.trange(int(n_iters))):
        if checkpoint > 0 and iter_idx > 0 and iter_idx % checkpoint == 0:
            checkpoint_region()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=dtype)
        logits = bound_forward_pass(model, generators, coeffs, inputs, use_zonotopes=True)
        margin = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            tau=tau,
            lower=True,
            aggregation="min",
            mode=multi_label_mode,
            surrogate=surrogate,
        )
        row_norms = torch.linalg.vector_norm(generators, dim=1).clamp_min(1e-12)
        size = torch.log(row_norms).mean()
        loss = -size + 100.0 * torch.relu(-margin).square()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            generators.clamp_(min=-float(max_generator_abs), max=float(max_generator_abs))
        pbar.set_postfix({"size": f"{float(size.detach().item()):.3f}", "margin": f"{float(margin.detach().item()):.3f}"})

    checkpoint_region()
    return ZonotopeRashomonResult(
        regions=regions,
        certificates=certificates,
        temperatures=temperatures,
        surrogate=surrogate,
        resolved_surrogate=resolved_surrogate,
    )


def select_certified_zonotope(result: ZonotopeRashomonResult) -> tuple[ZonotopeRegion, int] | None:
    """Select the last fully certified zonotope checkpoint."""

    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in result.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        return None
    idx = valid_indices[-1]
    return result.regions[idx], int(idx)
