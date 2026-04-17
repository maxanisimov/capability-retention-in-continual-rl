from src.trainer import SimpleTrainer
from src.regulariser import BaseRegulariser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
import copy


class LwFTrainer(SimpleTrainer):
    def __init__(
        self,
        model: nn.Module,
        context_sets: list[torch.Tensor],
        seed: int = 42,
        paradigm: str = "TIL",
        domain_map_fn: Callable = None,
        lmbd: float = 0.3,
        temperature: float = 2.0,
        **kwargs: dict,
    ):
        super().__init__(
            model, seed=seed, paradigm=paradigm, domain_map_fn=domain_map_fn
        )

        self.lmbd = lmbd
        self.temperature = temperature
        self.previous_task_models = {}
        self.previous_task_models[0] = copy.deepcopy(model)
        self.context_sets = context_sets

    def compute_distillation_loss(
        self, outputs: torch.Tensor, inputs: torch.Tensor
    ) -> None:
        distill_loss = 0.0
        if not self.previous_task_models:
            return distill_loss

        for task, old_model in self.previous_task_models.items():
            with torch.no_grad():
                old_logits = old_model(inputs)

            old_probs = F.softmax(old_logits / self.temperature, dim=1)
            new_log_probs = F.log_softmax(outputs / self.temperature, dim=1)

            task_classes = self.context_sets[task]
            if self.domain_map_fn is not None:
                task_classes = self.domain_map_fn(torch.tensor(task_classes)).tolist()

            for o in task_classes:
                distill_loss -= torch.sum(old_probs[:, o] * new_log_probs[:, o])

        return distill_loss * self.lmbd

    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        context_id: int = None,
        **kwargs: dict,
    ) -> nn.Module:
        assert context_id is not None, "context_id needs to be provided"

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None
        stopping = False

        if self.paradigm == "TIL":
            self._set_context(model, context_id=context_id)

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                def distillation_loss_fn(
                    predictions: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
                    loss = loss_fn(predictions, targets)
                    return loss + self.compute_distillation_loss(predictions, inputs)

                loss, _ = self._train_step(
                    model,
                    inputs,
                    targets,
                    optimizer,
                    distillation_loss_fn,
                    regulariser,
                    context_id=context_id if self.paradigm == "TIL" else None,
                )

                if max(1, i) % kwargs.get("val_freq", 100) == 0:
                    val_loss, val_acc = self._validate_model(
                        model,
                        val_loader,
                        loss_fn,
                        context_id if self.paradigm == "TIL" else None,
                    )

                    pbar.set_postfix(
                        {
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc:.4f}"
                            if val_acc is not None
                            else None,
                        }
                    )

                    if early_stopping:
                        stopping, epochs_no_improve = self.check_early_stopping(
                            curr_loss=val_loss,
                            prev_loss=best_val_loss
                            if best_val_loss != float("inf")
                            else None,
                            patience=kwargs.get("patience", 5),
                            no_improvement=epochs_no_improve,
                        )

                        best_val_loss = min(val_loss, best_val_loss)

                        if stopping:
                            print(f"Early stopping at epoch {epoch + 1}")
                            if best_model_state is not None:
                                model.load_state_dict(best_model_state)
                            break
                        elif val_loss == best_val_loss:
                            best_model_state = model.state_dict()
            if stopping:
                break

        self.previous_task_models[context_id] = copy.deepcopy(model)

        return model
