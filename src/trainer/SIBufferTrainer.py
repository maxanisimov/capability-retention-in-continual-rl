from src.trainer import SITrainer
from src.buffer import Buffer
from src.regulariser import BaseRegulariser

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Optimizer
from typing import Callable


class SIBufferTrainer(SITrainer):
    def __init__(
        self,
        model: nn.Module,
        buffer: Buffer,
        domain_map_fn: Callable = None,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(
            model=model, seed=seed, domain_map_fn=domain_map_fn, **rashomon_kwargs
        )
        self.buffer = buffer

    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ) -> nn.Module:
        """
        Train the model until either reaching the target accuracy or plateuing.
        If target accuracy reached -> continue until plateu and then exit
        Elif buffer has data -> recompute bounds
        Else -> Exit without full convergence
        """
        target_acc = kwargs.get("target_acc", None)
        assert target_acc is not None, (
            "Target accuracy must be provided for BufferTrainer."
        )
        MAX_BUFFER_CALLS = kwargs.get("max_buffer_calls", 3)
        buffer_calls = 0

        kwargs["patience"] = 5  # overwrite the patience used to always be 5
        while True:
            # Train model until early stopping is triggered
            # i.e. best performance has been extracted given current constraints
            super()._train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=100,
                early_stopping=True,
                regulariser=regulariser,
                **kwargs,
            )

            _, val_acc = self._validate_model(
                model,
                val_loader,
                loss_fn,
                regulariser,
                kwargs.get("context_id", None),
            )
            # If the extracted performance is greater than target performance -> exit
            if val_acc > target_acc:
                break
            # If performance not greater than target performance -> attempt recompute rashomon set
            elif not self.buffer.is_empty() and buffer_calls < MAX_BUFFER_CALLS:
                print("Consume buffer data.")
                (buffer_X, buffer_y), _ = self.buffer.consume()
                samples = next(iter(train_loader))
                self.compute_rashomon_set(
                    TensorDataset(buffer_X, buffer_y),
                    context_id=self.rashomon_kwargs.get("context_id", None),
                    use_outer_bbox=False,
                    si_batch=samples,
                    si_steps=kwargs.get("si_steps", 50),
                    prune_prop=kwargs.get("prune_prop", 0.8),
                )
                buffer_calls += 1
            else:
                break

        return model

    def add_to_buffer(self, dataset: Dataset) -> None:
        self.buffer.add_data(dataset)
