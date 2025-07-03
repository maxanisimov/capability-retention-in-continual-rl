from src.trainer.IntervalTrainer import IntervalTrainer
from src.buffer import Buffer
from src.regulariser.BaseRegulariser import BaseRegulariser

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Optimizer
from typing import Callable
from tqdm import tqdm


class BufferTrainer(IntervalTrainer):
    def __init__(
        self, model: nn.Module, buffer: Buffer, seed: int = 42, **rashomon_kwargs: dict
    ):
        super().__init__(model=model, seed=seed, **rashomon_kwargs)
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

        best_loss = float("inf")

        _running = True
        pbar = tqdm(desc="Processing items")
        no_improvement = 0
        buffer_no_improvement = 0
        while _running:
            pbar.update(1)
            for inputs, targets in train_loader:
                loss, acc = super()._train_step(
                    model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                )

                pbar.set_postfix(
                    {
                        "Training loss": loss,
                        "Training accuracy": acc,
                        "Buffer Data Consumed": self.buffer.samples_consumed(),
                    }
                )

                stop, no_improvement = super().check_early_stopping(
                    curr_loss=loss,
                    prev_loss=best_loss,
                    patience=kwargs.get("patience", 10),
                    no_improvement=no_improvement,
                )
                if not no_improvement:
                    buffer_no_improvement = 0
                best_loss = min(loss, best_loss)
                if stop:
                    if (
                        not self.buffer.is_empty()
                        and acc < target_acc
                        and buffer_no_improvement < 3
                    ):
                        print("Using buffer to continue training.")
                        (buffer_X, buffer_y), _ = self.buffer.consume()
                        self.compute_rashomon_set(
                            TensorDataset(buffer_X, buffer_y),
                            context_id=kwargs.get("context_id", None),
                            use_outer_bbox=False,
                        )
                        no_improvement = 0  # Reset early stopping mechanism
                        buffer_no_improvement += 1
                    else:
                        print(f"Exiting with final training accuracy of {acc:.4f}")
                        _running = False
                        break

        return model

    def add_to_buffer(self, dataset: Dataset) -> None:
        self.buffer.add_data(dataset)
