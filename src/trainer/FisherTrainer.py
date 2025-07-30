from src.trainer import IntervalTrainer
from typing import Callable

import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class FisherTrainer(IntervalTrainer):
    def __init__(
        self,
        model: nn.Module,
        projection_strategy: str = "closest",
        n_certificate_samples=256,
        min_acc_increment=0.05,
        min_acc_limit=0.9,
        seed: int = 42,
        paradigm: str = "TIL",
        **rashomon_kwargs: dict,
    ):
        super().__init__(
            model,
            projection_strategy=projection_strategy,
            n_certificate_samples=n_certificate_samples,
            min_acc_increment=min_acc_increment,
            min_acc_limit=min_acc_limit,
            seed=seed,
            paradigm=paradigm,
            **rashomon_kwargs,
        )

        self.fisher_matrix = None

    def _get_mask(
        self, fisher_diagonal: list[torch.Tensor], percentage: float = 0.3
    ) -> list[torch.Tensor]:
        all_scores = torch.cat([scores.view(-1) for scores in fisher_diagonal])

        if all_scores.numel() == 0:
            return [torch.zeros_like(s, dtype=torch.bool) for s in fisher_diagonal]

        print("Parameter importance sum:", (torch.sum(all_scores).item()))
        print("Non zero params:", torch.count_nonzero(all_scores).item())
        prune_fraction = percentage
        threshold = torch.quantile(all_scores, prune_fraction)

        print(
            f"To keep top {1 - percentage:.0%}, found global Fisher threshold: {threshold.item():.8f}"
        )

        pruning_masks = []
        for scores in fisher_diagonal:
            # True -> mask out param, False -> do not mask out
            mask = scores < threshold
            pruning_masks.append(mask)

        print("Remaining number of parameters:", sum([torch.count_nonzero(mask) for mask in pruning_masks]))

        return pruning_masks

    def _compute_fisher_diagonal(
        self, batch: tuple[torch.Tensor, torch.Tensor], epochs: int = 10
    ) -> None:
        fisher = [
            torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad
        ]
        self.model.eval()
        num_samples = 0

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if self.domain_map_fn:
            targets = self.domain_map_fn(targets)
        for _ in range(epochs):
            num_samples += inputs.size(0)
            outputs = self.model(inputs)

            log_likelihood = F.log_softmax(outputs, dim=1)
            log_likelihood_for_targets = log_likelihood[range(len(targets)), targets]

            for ll in log_likelihood_for_targets:
                self.model.zero_grad()
                ll.backward(retain_graph=True)
                for i, p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        fisher[i] += p.grad.data.pow(2)

        self.fisher_matrix = [f / num_samples for f in fisher]
        self.model.train()

    def compute_rashomon_set(
        self,
        dataset: Dataset,
        next_task_data: Dataset,
        prune_prop: float = 0.3,
        callback: Callable = None,
        use_outer_bbox: bool = True,
        context_id: int = None,
        **kwargs: dict,
    ) -> None:
        loader = DataLoader(
            next_task_data,
            batch_size=kwargs.get("fisher_batch_size", 64),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self._compute_fisher_diagonal(
            next(iter(loader)), epochs=kwargs.get("fisher_epochs", None)
        )
        mask = (
            self._get_mask(self.fisher_matrix, prune_prop)
            if self.fisher_matrix
            else None
        )
        return super().compute_rashomon_set(
            dataset, callback, use_outer_bbox, context_id, param_mask=mask
        )
