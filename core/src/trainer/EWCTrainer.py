from src.trainer import SimpleTrainer
from src.regulariser import BaseRegulariser
from typing import Callable

import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class EWCTrainer(SimpleTrainer):
    def __init__(
        self,
        model: nn.Module,
        seed: int = 42,
        paradigm: str = "TIL",
        domain_map_fn: Callable = None,
        lmbd: float = 0.3,
        **kwargs: dict
    ):
        super().__init__(
            model, seed=seed, paradigm=paradigm, domain_map_fn=domain_map_fn
        )

        self.fisher_at_task = []
        self.params_at_task = []
        self.lmbd = lmbd

    def compute_fisher_information(
        self,
        model: nn.Module,
        dataset: Dataset,
        loss_fn: Callable,
        context_id: int = None,
        batch_size: int = 32,
    ) -> None:
        X, y = next(
            iter(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(self.seed)
                    if self.seed is not None
                    else None,
                )
            )
        )
        X, y = X.to(self.device), y.to(self.device)
        if self.domain_map_fn is not None:
            y = self.domain_map_fn(y)
        self._set_context(model, context_id=context_id)

        parameters = list(model.parameters())

        for p in parameters:
            p.requires_grad = True
            p.grad = None

        logits = model.forward(X)
        loss = loss_fn(logits, y)
        loss.backward()
        fisher = [p.grad.pow(2).detach() for p in parameters]

        params = [p.clone().detach() for p in model.parameters()]

        self.fisher_at_task.append(fisher)
        self.params_at_task.append(params)

    def _train_step(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        regulariser: BaseRegulariser = None,
        context_id: int = None,
        **kwargs: dict,
    ) -> tuple[float, float]:
        model.train()
        inputs, targets = X.to(self.device), y.to(self.device)

        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            targets = self.domain_map_fn(targets)
        self._set_context(model, context_id)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        for task_fisher, task_params in zip(self.fisher_at_task, self.params_at_task):
            for p, f, pp in zip(model.parameters(), task_fisher, task_params):
                loss += (self.lmbd / 2) * (f * (p - pp).pow(2)).sum()
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == targets).sum() / len(targets)

        return loss, acc

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
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None
        stopping = False

        self._set_context(model, kwargs.get("context_id", None))

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                loss, _ = self._train_step(
                    model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                )

                if max(1, i) % kwargs.get("val_freq", 100) == 0:
                    val_loss, val_acc = self._validate_model(
                        model,
                        val_loader,
                        loss_fn,
                        kwargs.get("context_id", None),
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

        self.compute_fisher_information(
            model,
            train_loader.dataset,
            loss_fn,
            context_id=kwargs.get('context_id', None),
            batch_size=kwargs.get("fisher_batch", 32),
        )

        return model
