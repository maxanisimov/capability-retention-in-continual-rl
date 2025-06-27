from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.utils import print_bold

from abc import ABC, abstractmethod
from typing import Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import copy


class BaseTrainer(ABC):
    def __init__(self, model: nn.Module | IntervalBoundedModel, seed: int = 42):
        self.seed = seed
        self.model = copy.deepcopy(model)
        self.device = (
            next(model.parameters()).device
            if isinstance(model, nn.Module)
            else next(model.param_l).device
        )
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "mps":
            torch.mps.manual_seed(seed)
        elif self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(
        self,
        train_data: Dataset,
        val_data: Dataset,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ) -> None:
        self.model.train()
        train_loader = DataLoader(
            train_data,
            batch_size=kwargs.get("batch_size", 32),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_data, batch_size=kwargs.get("batch_size", 32), shuffle=False
        )
        optimizer = kwargs.get(
            "optimizer",
            torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get("lr", 0.001),
                weight_decay=kwargs.get("weight_decay", 0),
            ),
        )
        loss_fn = kwargs.get("loss_fn", nn.CrossEntropyLoss())
        self.model = self._train(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            early_stopping=early_stopping,
            regulariser=regulariser,
            **kwargs,
        )

    @abstractmethod
    def _train(
        model: nn.Module | IntervalBoundedModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: "BaseRegulariser" = None,
        **kwargs: dict,
    ) -> nn.Module | IntervalBoundedModel:
        pass

    @torch.no_grad()
    def test(self, test_data: list[Dataset], **kwargs: dict) -> None:
        self.model.eval()
        loss_fn = kwargs.get("loss_fn", nn.CrossEntropyLoss())

        outputs = []
        for i, (task_data, context_id) in enumerate(
            zip(test_data, kwargs.get("context_list", [None] * len(test_data)))
        ):
            test_loader = DataLoader(task_data, batch_size=1028, shuffle=False)
            outputs.append(
                self._test(
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    context_id=context_id,
                    **kwargs,
                )
            )

        print_bold(f"Test Results: {outputs}")
        return outputs

    @abstractmethod
    @torch.no_grad()
    def _test(
        self, test_loader: DataLoader, loss_fn: Callable, **kwargs: dict
    ) -> tuple[float, float]:
        pass

    def check_early_stopping(
        self,
        curr_loss: float,
        prev_loss: float = None,
        patience: int = 5,
        no_improvement: int = 0,
    ) -> Tuple[bool, int]:
        if prev_loss is None or curr_loss < prev_loss:
            return False, 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                return True, no_improvement
            else:
                return False, no_improvement

    @torch.no_grad()
    def _validate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Callable,
        regulariser: BaseRegulariser = None,
        domain_map_fn: Callable = None,
    ) -> tuple[float, float]:
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if domain_map_fn:
                    targets = domain_map_fn(targets)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if regulariser is not None:
                    loss += regulariser(model=model, inputs=inputs)
                val_loss += loss.item()
                correct += (outputs.argmax(dim=1) == targets).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        return val_loss, val_acc
