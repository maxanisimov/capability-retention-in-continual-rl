from src.trainer.BaseTrainer import BaseTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm


class SimpleTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        paradigm: str = None,
        domain_map_fn: Callable = None,
        seed: int = 42,
    ):
        super().__init__(
            model=model, paradigm=paradigm, domain_map_fn=domain_map_fn, seed=seed
        )

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
        stopping = False

        pbar = tqdm(range(epochs), desc="Training Epochs", postfix={"val_loss": "N\A"})
        for epoch in pbar:
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if hasattr(self, "domain_map_fn") and self.domain_map_fn:
                    targets = self.domain_map_fn(targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if regulariser is not None:
                    loss += regulariser(model=model, inputs=inputs, targets=targets)
                loss.backward()
                optimizer.step()

                if i % kwargs.get("val_freq", 10):
                    val_loss, val_acc = self._validate_model(
                        model,
                        val_loader,
                        loss_fn,
                        regulariser,
                        context_id=kwargs.get("context_id", None),
                    )

                    pbar.set_postfix({"val_loss": val_loss, "val_acc": val_acc})

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
                        else:
                            best_model_state = model.state_dict()
            if stopping:
                break

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

            if hasattr(self, "domain_map_fn") and self.domain_map_fn:
                targets = self.domain_map_fn(targets)

            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        return round(avg_loss, 4), round(accuracy, 4)
