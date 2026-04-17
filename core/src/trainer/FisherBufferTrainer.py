from src.trainer import FisherTrainer
from src.buffer import MultiTaskBuffer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FisherBufferTrainer(FisherTrainer):
    def __init__(
        self,
        model: nn.Module,
        buffer: MultiTaskBuffer,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(model=model, seed=seed, **rashomon_kwargs)
        self.buffer = buffer

        self.task_bounds = []
        self.task_certificates = []
        self.recall_dataset = None
        self.recall_iter = None

    def train(
        self,
        train_data,
        val_data,
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
        return super().train(
            train_data, val_data, epochs, early_stopping, regulariser, **kwargs
        )

    def _train_step(
        self,
        model,
        X,
        y,
        optimizer,
        loss_fn,
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
            model, X, y, optimizer, loss_fn, regulariser, project, context_id, **kwargs
        )

    def add_to_buffer(self, dataset: Dataset, task_id: int, k: int) -> None:
        self.buffer.add_data(dataset, task_id, k=k)
