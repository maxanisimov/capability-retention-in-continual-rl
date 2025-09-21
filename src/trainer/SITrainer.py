from src.trainer import IntervalTrainer
from src.helpers.SITracker import SITracker

import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Callable
from copy import deepcopy


class SITrainer(IntervalTrainer):
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

    def _get_mask(
        self, si_scores: list[torch.Tensor], percentage: float = 0.3, preserve_top: bool = True
    ) -> list[torch.Tensor]:
        all_scores = torch.cat([scores.view(-1) for scores in si_scores])

        if all_scores.numel() == 0:
            return [torch.zeros_like(s, dtype=torch.bool) for s in si_scores]

        prune_fraction = percentage
        threshold = torch.quantile(all_scores, prune_fraction)

        print(
            f"Found global SI threshold: {threshold.item():.8f}"
        )

        pruning_masks = []
        for scores in si_scores:
            # True -> freeze parameter, False -> do not freeze parameter
            mask = scores > threshold if preserve_top else scores < threshold
            pruning_masks.append(mask)

        count_true = sum([torch.sum(mask).item() for mask in pruning_masks])
        count_total = sum([mask.numel() for mask in pruning_masks])
        print(f"Freezing {'MOST' if preserve_top else 'LEAST'} important {count_true} out of {count_total} parameters.")

        return pruning_masks

    def compute_rashomon_set(
        self,
        dataset: Dataset,
        callback: Callable = None,
        use_outer_bbox: bool = True,
        context_id: int = None,
        **kwargs: dict,
    ) -> None:
        prune_prop = kwargs.get("prune_prop", None)
        si_batch = kwargs.get("si_batch", None)
        si_steps = kwargs.get("si_steps", 50)

        assert prune_prop is not None, (
            "prune_prop required for SITrainer.compute_rashomon_set"
        )
        assert si_batch is not None or prune_prop == 0, (
            "si_batch required for SITrainer.compute_rashomon_set when prun_prop > 0"
        )

        if self.paradigm == "TIL":
            assert context_id is not None, (
                "context_id required for SITrainer.compute_rashomon_set for TIL"
            )
            assert "si_context_id" in kwargs or prune_prop == 0, (
                "si_context_id required for SITrainer.compute_rashomon_set for TIL"
            )

        mask = None
        if prune_prop > 0:
            temp_model = deepcopy(self.model)
            optimizer = torch.optim.Adam(
                temp_model.parameters(), lr=kwargs.get("si_lr", 0.001)
            )
            loss_fn = nn.CrossEntropyLoss()

            si = SITracker(temp_model)
            temp_model.train()
            for _ in range(si_steps):
                inputs, targets = si_batch
                loss, _ = self._train_step(
                    temp_model,
                    inputs,
                    targets,
                    optimizer,
                    loss_fn,
                    project=False,
                    context_id=kwargs.get("si_context_id", None),
                )
                si.update_importance()

            mask = self._get_mask(si.importance_scores, prune_prop, kwargs.get("preserve_top", False))
        return super().compute_rashomon_set(
            dataset, callback, use_outer_bbox, context_id, param_mask=mask
        )
