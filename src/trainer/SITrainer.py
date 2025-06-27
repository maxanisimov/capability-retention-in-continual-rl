from src.trainer import IntervalTrainer
from src.helpers.SITracker import SITracker
import src.utils as utils
from src.regulariser import BaseRegulariser
from typing import Callable

import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


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

        self.si = SITracker(self.model)

    def _train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ) -> nn.Module:
        assert self.bounds, "Bounds must have been computed prior to calling train."

        self._project_parameters(model, self.bounds)

        best_train_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None
        stopping = False

        if (
            self.paradigm == "TIL"
            and (context := kwargs.get("context_id", None)) is not None
            and isinstance(model[-1], utils.InContextHead)
        ):
            model[-1].set_context(context)

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            for inputs, targets in train_loader:
                loss, _ = self._train_step(
                    model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                )

                # Track synaptic intelligence
                self.si.update_importance()

                if early_stopping:
                    stopping, epochs_no_improve = self.check_early_stopping(
                        curr_loss=loss,
                        prev_loss=best_train_loss
                        if best_train_loss != float("inf")
                        else None,
                        patience=kwargs.get("patience", 5),
                        no_improvement=epochs_no_improve,
                    )

                    best_train_loss = min(best_train_loss, loss)

                    if stopping:
                        print(f"Early stopping at epoch {epoch + 1}")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
                    else:
                        best_model_state = model.state_dict()

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "val_acc": f"{val_acc:.4f}" if val_acc is not None else None,
                    "proj": self._last_projection,
                }
            )

            val_loss, val_acc = self._validate_model(
                model,
                val_loader,
                loss_fn,
                regulariser,
                domain_map_fn=self.domain_map_fn
                if hasattr(self, "domain_map_fn")
                else None,
            )

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "val_acc": f"{val_acc:.4f}" if val_acc is not None else None,
                    "proj": self._last_projection,
                }
            )

            if stopping:
                break

        return model

    def _get_mask(
        self, si_scores: list[torch.Tensor], percentage: float = 0.3
    ) -> list[torch.Tensor]:
        all_scores = torch.cat([scores.view(-1) for scores in si_scores])

        if all_scores.numel() == 0:
            return [torch.zeros_like(s, dtype=torch.bool) for s in si_scores]

        prune_fraction = percentage
        threshold = torch.quantile(all_scores, prune_fraction)

        print(
            f"To keep top {1 - percentage:.0%}, found global SI threshold: {threshold.item():.4f}"
        )

        pruning_masks = []
        for scores in si_scores:
            # True -> mask out param, False -> do not mask out
            mask = scores < threshold
            pruning_masks.append(mask)

        return pruning_masks

    def compute_rashomon_set(
        self,
        dataset: Dataset,
        prune_prop: float = 0.3,
        callback: Callable = None,
        use_outer_bbox: bool = True,
        context_id: int = None,
    ) -> None:
        mask = self._get_mask(self.si.importance_scores, prune_prop)
        return super().compute_rashomon_set(
            dataset, callback, use_outer_bbox, context_id, param_mask=mask
        )
