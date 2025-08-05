from src.trainer import BaseSmoothTrainer

import torch
import torch.nn as nn
from typing import Callable
from tqdm import trange
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm


class LoRASmoothTrainer(BaseSmoothTrainer):
    def __init__(
        self,
        model: nn.Module,
        peft_model: nn.Module,
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

        self.peft_model = peft_model

    def init_density(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        metric: Callable,
        bound: float,
        iterations: int,
        delta: float,
        **kwargs: dict,
    ):
        X, y = batch[0].to(self.device), batch[1].to(self.device)
        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            y = self.domain_map_fn(y)
        
        self.mu = []
        self.sigma = []
        lora_modules = []

        for module in self.peft_model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_modules.append((module, module.lora_A.default.weight, module.lora_B.default.weight))
                self.mu.append(module.lora_A.default.weight.detach().clone())
                self.mu.append(module.lora_B.default.weight.detach().clone())

                eps = kwargs.get("eps", 0.01)
                self.sigma.append(torch.full_like(module.lora_A.default.weight, eps))
                self.sigma.append(torch.full_like(module.lora_B.default.weight, eps))

        max_rad = -1e6
        for inflate in self.inflates:
            success_count = 0

            for _ in trange(iterations):
                new_weights = []
                for mu, sigma in zip(self.mu, self.sigma):
                    scale_tensor = sigma * inflate
                    noise = torch.normal(mean=0.0, std=scale_tensor).to(self.device)
                    new_weights.append(mu + noise)

                with torch.no_grad():
                    idx = 0
                    for module, A, B in lora_modules:
                        A.copy_(new_weights[idx])
                        B.copy_(new_weights[idx+1])
                        idx += 2

                
                    perturbed_acc = metric(X, y, self.model)

                    if perturbed_acc >= bound:
                        success_count += 1

            with torch.no_grad():
                idx = 0
                for module, A, B in lora_modules:
                    A.copy_(self.mu[idx])
                    B.copy_(self.mu[idx + 1])
                    idx += 2

            lower_bound, _ = proportion_confint(
                success_count * self.smooth_cheat,
                iterations * self.smooth_cheat,
                alpha=delta,
                method='beta'
            )

            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate

            print(f"Inflate: {inflate:.1f}, Estimated success: {success_count/iterations:.4f}, Certified radius: {certified_radius:.4f}")

            if certified_radius > max_rad:
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break

        print(f"[LoRA] Max certified radius: {(0.01 * max_inflate) * max_rad:.4f} at inflate {max_inflate}")

        idx = 0
        for module, A, B in lora_modules:
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                module.A0 = module.lora_A.default.weight.detach().clone()
                module.B0 = module.lora_B.default.weight.detach().clone()

            idx += 2

        self.sigma = [scale * max_inflate for scale in self.sigma]
        self.alpha = max_rad * (0.01 * max_inflate)

        return max_rad
