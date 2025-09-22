from src.trainer import BaseSmoothTrainer
from abstract_gradient_training.bounded_models import IntervalBoundedModel

from typing import Callable
import torch
import torch.nn as nn
from tqdm import trange
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


class SmoothTrainer(BaseSmoothTrainer):
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
        bounded_model: IntervalBoundedModel = None,
        **kwargs,
    ):
        """
        Initializes self.mu and self.scale from interval bounds and certifies the smoothing radius.
        """

        assert bounded_model, (
            "SmoothTrainer.init_density requires bounded_model to be set."
        )
        X, y = batch[0].to(self.device), batch[1].to(self.device)
        if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
            y = self.domain_map_fn(y)
        
        max_rad = -1e6

        self.mu = [
            p.detach().clone().to(torch.float32).to(self.device)
            for p in self.model.parameters()
        ]
        widths = [
            (param_u - param_l).to(torch.float32).to(self.device)
            for param_l, param_u in zip(bounded_model.param_l, bounded_model.param_u)
        ]

        for inflate in self.inflates:

            test_scale = [w * inflate for w in widths]

            # Step 3: Assert nominal accuracy
            acc = metric(X, y, self.model)
            assert acc >= bound, (
                f"Nominal accuracy {acc:.4f} does not exceed threshold {bound:.4f}"
            )

            # Step 4: Run smoothing simulations
            success_count = 0
            for _ in trange(iterations):
                new_params = []
                for loc, scale in zip(self.mu, test_scale):
                    noise = torch.normal(mean=0.0, std=scale)
                    new_params.append((loc + noise).reshape(loc.shape))

                with torch.no_grad():
                    for p, new_p in zip(self.model.parameters(), new_params):
                        p.copy_(new_p)

                perturbed_acc = metric(X, y, self.model)
                if perturbed_acc >= bound:
                    success_count += 1

            # Step 5: Restore nominal parameters
            with torch.no_grad():
                for p, loc in zip(self.model.parameters(), self.mu):
                    p.copy_(loc)

            # Step 6: Compute Clopper-Pearson bound
            lower_bound, _ = proportion_confint(
                success_count * self.smooth_cheat,
                iterations * self.smooth_cheat,
                alpha=delta,
                method="beta",
            )
            print(
                "Estimated success: ",
                success_count / iterations,
                " Lower bound: ",
                lower_bound,
            )
            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate

            print(f"Certified L2 radius: {certified_radius:.4f}")

            if certified_radius > max_rad:
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break

        print("Max rad: ", max_rad, " inflate: ", max_inflate)
        print("Last rad: ", certified_radius, " last inflate: ", inflate)
        self.sigma = [max_inflate * w for w in widths]
        self.alpha = max_rad
        return certified_radius
