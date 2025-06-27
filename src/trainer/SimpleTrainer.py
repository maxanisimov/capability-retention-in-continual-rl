from src.trainer.BaseTrainer import BaseTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm


class SimpleTrainer(BaseTrainer):
    def __init__(
        self, model: nn.Module, domain_map_fn: Callable = None, seed: int = 42
    ):
        super().__init__(model=model, seed=seed)
        self.domain_map_fn = domain_map_fn

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
        **kwargs: dict,
    ) -> nn.Module:
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        pbar = tqdm(range(epochs), desc="Training Epochs", postfix={"val_loss": "N\A"})
        for epoch in pbar:
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.domain_map_fn:
                    targets = self.domain_map_fn(targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if regulariser is not None:
                    loss += regulariser(model=model, inputs=inputs, targets=targets)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    if self.domain_map_fn:
                        targets = self.domain_map_fn(targets)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    if regulariser is not None:
                        loss += regulariser(model=model, inputs=inputs, targets=targets)
                    val_loss += loss.item()
                    correct += (outputs.argmax(dim=1) == targets).sum().item()
            val_loss /= len(val_loader)
            pbar.set_postfix(
                {"val_loss": val_loss, "val_acc": correct / len(val_loader.dataset)}
            )

            if early_stopping:
                stopping, epochs_no_improve = self.check_early_stopping(
                    curr_loss=val_loss,
                    prev_loss=best_val_loss if best_val_loss != float("inf") else None,
                    patience=kwargs.get("patience", 5),
                    no_improvement=epochs_no_improve,
                )

                if stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
                else:
                    best_model_state = model.state_dict()

        return model

    @torch.no_grad()
    def _test(
        self, test_loader: DataLoader, loss_fn: Callable, **kwargs: dict
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.domain_map_fn:
                targets = self.domain_map_fn(targets)

            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        return round(avg_loss, 4), round(accuracy, 4)
