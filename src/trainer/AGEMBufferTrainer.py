from src.trainer import IntervalAGEMTrainer
from src.buffer import MultiTaskBuffer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class AGEMBufferTrainer(IntervalAGEMTrainer):
    def __init__(
        self,
        model: nn.Module,
        buffer: MultiTaskBuffer,
        memory_samples: int = 200,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(model=model, seed=seed, **rashomon_kwargs)
        self.buffer = buffer

        self.task_bounds = []
        self.task_certificates = []
        self.recall_dataset = None
        self.recall_iter = None
        self.memory_samples = memory_samples
        self.memory_batches = []

    def add_to_buffer(self, dataset: Dataset, task_id: int, k: int = 200) -> None:
        self.buffer.add_data(dataset, task_id, k=k)

    def remove_oldest_buffer(self) -> None:
        if isinstance(self.buffer, MultiTaskBuffer):
            self.buffer.drop_buffer(0)

    def _train(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        epochs=5,
        early_stopping=False,
        regulariser=None,
        **kwargs,
    ):
        batch_size = kwargs.get("batch_size", 32)
        if self.recall_dataset:
            self.recall_iter = iter(
                DataLoader(
                    self.recall_dataset,
                    batch_size=batch_size
                    if len(self.recall_dataset) >= batch_size
                    else len(self.recall_dataset),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(self.seed),
                )
            )

        return super()._train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            epochs,
            early_stopping,
            regulariser,
            **kwargs,
        )

    def _train_step(
        self,
        model,
        X,
        y,
        optimizer,
        loss_fn,
        memory_batches=[],
        regulariser=None,
        project=True,
        context_id=None,
        **kwargs,
    ):
        if self.recall_iter:
            recall_X, recall_y = next(self.recall_iter, (None, None))
            if recall_X is None:
                self.recall_iter = iter(
                    DataLoader(
                        self.recall_dataset,
                        batch_size=len(X)
                        if len(self.recall_dataset) >= len(X)
                        else len(self.recall_dataset),
                        shuffle=True,
                        generator=torch.Generator().manual_seed(self.seed),
                    )
                )
                recall_X, recall_y = next(self.recall_iter)
            X = torch.cat([X, recall_X], dim=0)
            y = torch.cat([y, recall_y], dim=0)

        return super()._train_step(
            model,
            X,
            y,
            optimizer,
            loss_fn,
            memory_batches,
            regulariser,
            project,
            context_id,
            **kwargs,
        )
