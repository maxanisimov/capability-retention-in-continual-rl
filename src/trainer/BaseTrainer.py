from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.utils import print_bold, InContextHead, print_colored

from abc import ABC, abstractmethod
from typing import Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import copy


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module | IntervalBoundedModel,
        paradigm: str = None,
        domain_map_fn: Callable = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.model = copy.deepcopy(model)
        self.device = (
            next(model.parameters()).device
            if isinstance(model, nn.Module)
            else next(model.param_l).device
        )
        self.set_seed(seed)
        self.paradigm = paradigm
        if paradigm == "DIL":
            assert domain_map_fn, "domain_map_fn required for DIL"
            self.domain_map_fn = domain_map_fn

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
        """
        Args:
            train_data (Dataset): The dataset used for training the model.
            val_data (Dataset): The dataset used for validating the model's
                performance after each epoch.
            epochs (int, optional): The total number of training epochs to run.
                Defaults to 5.
            early_stopping (bool, optional): If True, enables early stopping based
                on validation performance within the `_train` method. Defaults to False.
            regulariser (BaseRegulariser, optional): An optional regularisation
                object to be applied during the training loop. Defaults to None.
            **kwargs (dict): A dictionary of additional keyword arguments for
                customizing the training setup. Accepted keys include:
                - batch_size (int): The batch size for DataLoaders. Defaults to 32.
                - optimizer (torch.optim.Optimizer): A pre-configured optimizer.
                  If not provided, defaults to Adam with parameters from `lr` and
                  `weight_decay`.
                - lr (float): Learning rate for the default Adam optimizer.
                  Defaults to 0.001.
                - weight_decay (float): Weight decay for the default Adam optimizer.
                  Defaults to 0.
                - loss_fn (torch.nn.Module): The loss function to use.
                  Defaults to nn.CrossEntropyLoss.
                - context_id: Required when `self.paradigm` is "TIL". Used to
                  set the task context for the model.

        Returns:
            None. The method updates the internal model state (`self.model`) in place.

        Raises:
            AssertionError: If `self.paradigm` is "TIL" and `context_id` is not
                provided in `kwargs`.
        """
        if self.paradigm == "TIL":
            assert kwargs.get("context_id", None) is not None, (
                "context_id is required for TIL"
            )

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

        self._set_context(self.model, kwargs.get("context_id", None))
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
        """
        Args:
            test_data (list[Dataset]): A list of PyTorch Datasets to evaluate.
            **kwargs (dict): A dictionary of additional keyword arguments for
                customizing the testing setup. Accepted keys include:
                - loss_fn (torch.nn.Module): The loss function to use for
                  evaluation. Defaults to nn.CrossEntropyLoss.
                - context_list (list): A list of context identifiers, one for
                  each dataset in `test_data`. Required for the "TIL" paradigm.

        Returns:
            list: A list containing the evaluation results (e.g., a dictionary
            of metrics like loss and accuracy) for each dataset. The specific
            format of each element is determined by the `_test` method.

        Raises:
            AssertionError: If `self.paradigm` is "TIL" and `context_list` is not
                provided as a keyword argument.
        """
        if self.paradigm == "TIL":
            assert "context_list" in kwargs, "context_list required for TIL"
        self.model.eval()
        loss_fn = kwargs.get("loss_fn", nn.CrossEntropyLoss())

        outputs = []
        for i, (task_data, context_id) in enumerate(
            zip(test_data, kwargs.get("context_list", [None] * len(test_data)))
        ):
            test_loader = DataLoader(task_data, batch_size=1028, shuffle=False)
            self._set_context(self.model, context_id)
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
        context_id: int = None,
    ) -> tuple[float, float]:
        """
        Args:
            model (nn.Module): The model instance to be validated.
            val_loader (DataLoader): The DataLoader for the validation dataset.
            loss_fn (Callable): The loss function used to calculate the validation loss.
            regulariser (BaseRegulariser, optional): An optional regularisation
                penalty to add to the loss. Defaults to None.
            context_id (int, optional): The context identifier, which is required
                for the Task-Incremental Learning ("TIL") paradigm. Defaults to None.

        Returns:
            tuple[float, float]: A tuple containing the average validation loss
            and the overall validation accuracy.

        Raises:
            AssertionError: If `self.paradigm` is "TIL" and `context_id` is not provided.
        """
        # Validation phase
        model.eval()
        if self.paradigm == "TIL":
            assert context_id is not None, "context_id is required for TIL"
        self._set_context(model, context_id)

        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    targets = self.domain_map_fn(targets)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if regulariser is not None:
                    loss += regulariser(model=model, inputs=inputs)
                val_loss += loss.item()
                correct += (outputs.argmax(dim=1) == targets).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        return val_loss, val_acc

    def _set_context(self, model: nn.Module, context_id: int) -> None:
        if self.paradigm is None:
            return
        if context_id is None and isinstance(model[-1], InContextHead):
            print_colored(
                "Failed to set context: No context_id provided.", color="amber"
            )
        if context_id is not None and not isinstance(model[-1], InContextHead):
            print_colored(
                "Failed to set context: Last model layer is not a context head.",
                color="amber",
            )
        if isinstance(model[-1], InContextHead) and context_id is not None:
            model[-1].set_context(context_id)
