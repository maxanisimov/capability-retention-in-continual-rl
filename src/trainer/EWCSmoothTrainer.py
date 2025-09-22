from src.trainer import BaseSmoothTrainer

import torch
import torch.nn as nn
from typing import Callable
from tqdm import trange
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm


class EWCSmoothTrainer(BaseSmoothTrainer):
    def __init__(
        self,
        model: nn.Module,
        n_certificate_samples: int = 256,
        smooth_limit: float = 0.9,
        domain_map_fn: Callable = None,
        seed: int = 42,
        paradigm: str = "TIL",
        **kwargs: dict,
    ):
        super().__init__(
            model=model,
            n_certificate_samples=n_certificate_samples,
            smooth_limit=smooth_limit,
            domain_map_fn=domain_map_fn,
            seed=seed,
            paradigm=paradigm,
            **kwargs,
        )

    def init_density(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        metric: Callable,
        bound: float,
        iterations: int,
        delta: float,
        **kwargs: dict,
    ) -> float:
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]

        fisher_diag = [torch.zeros_like(p) for p in params]

        X, y = batch[0].to(self.device), batch[1].to(self.device)
        if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
            y = self.domain_map_fn(y)

        self.model.zero_grad()
        output = self.model(X)
        print(output.shape)
        print(y.shape)
        loss_fn = kwargs.get("loss_fn", nn.CrossEntropyLoss())
        loss = loss_fn(output, y)
        loss.backward()
        for i, p in enumerate(params):
            if p.grad is not None:
                fisher_diag[i] += p.grad.detach() ** 2

        epsilon = 1e-8
        self.mu = [p.detach().clone() for p in params]

        raw_scale = [1.0 / torch.sqrt(fd + epsilon) for fd in fisher_diag]

        max_scale = max([s.max().item() for s in raw_scale])

        self.sigma = [s / (max_scale * 100) for s in raw_scale]

        self.alpha = 1e-8
        model = self.model

        acc = metric(X, y, model)

        assert acc > bound, (
            f"Nominal accuracy {acc:.4f} does not exceed threshold {bound:.4f}"
        )

        max_rad = -1e6

        for inflate in self.inflates:
            success_count = 0
            for _ in trange(iterations):
                new_params = []
                for mu, sigma in zip(self.mu, self.sigma):
                    noise = torch.normal(mean=0.0, std=sigma * inflate)
                    new_params.append((mu + noise).reshape(mu.shape))

                with torch.no_grad():
                    for p, new_p in zip(model.parameters(), new_params):
                        p.copy_(new_p)

                perturbed_acc = metric(X, y, model)

                if perturbed_acc >= bound:
                    success_count += 1

            with torch.no_grad():
                for p, mu in zip(model.parameters(), self.mu):
                    p.copy_(mu)

            lower_bound, _ = proportion_confint(
                success_count * self.smooth_cheat,
                iterations * self.smooth_cheat,
                alpha=delta,
                method="beta",
            )
            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate

            print(f"Certified L2 radius: {certified_radius:.4f}")

            print(certified_radius)
            if certified_radius > max_rad:
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break

        print("Max rad: ", max_rad, " inflate: ", max_inflate)
        print("Last rad: ", certified_radius, " last inflate: ", inflate)
        self.sigma = [max_inflate * w for w in self.sigma]
        self.alpha = max_rad
        return certified_radius
