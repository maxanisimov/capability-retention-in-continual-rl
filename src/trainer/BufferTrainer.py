from src.trainer.IntervalTrainer import IntervalTrainer
from src.buffer import MultiTaskBuffer
from src.data_utils import get_mnist_tasks
from src import interval_utils

from abstract_gradient_training.bounded_models import IntervalBoundedModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import copy
import itertools


class BufferTrainer(IntervalTrainer):
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
        self.task_bounds_intersection = None
        self.intersection_certificates = None
        self.recall_dataset = None
        self.recall_iter = None

    def get_largest_bounds(self, k: int = 2) -> tuple[IntervalBoundedModel, float]:
        sizes = torch.tensor(
            [interval_utils._bounded_model_width(bm) for bm in self.bounds]
        )

        largest_bounds = torch.topk(sizes, k)[1]
        return [self.bounds[i] for i in largest_bounds], [
            self.certificates[i] for i in largest_bounds
        ]

    def add_to_buffer(self, dataset: Dataset, task_id: int, k: int = 200) -> None:
        self.buffer.add_data(dataset, task_id, k=k)

    def remove_oldest_buffer(self) -> None:
        if isinstance(self.buffer, MultiTaskBuffer):
            self.buffer.drop_buffer(0)

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

    ### Below methods only needed if bounds are split to be updated individually
    def _generate_intersection(
        self, bounds: list[IntervalBoundedModel]
    ) -> IntervalBoundedModel:
        train, val, test = get_mnist_tasks(seed=self.seed)

        intersect = copy.deepcopy(bounds[0])
        for bound in bounds[1:]:
            for i, (curr, new) in enumerate(zip(intersect._param_l, bound._param_l)):
                if not curr or not new:
                    continue
                curr_w, curr_b = curr
                new_w, new_b = new
                inter_w = torch.max(curr_w, new_w)
                inter_b = torch.max(curr_b, new_b)

                intersect._param_l[i] = [inter_w, inter_b]

            for i, (curr, new) in enumerate(zip(intersect._param_u, bound._param_u)):
                if not curr or not new:
                    continue
                curr_w, curr_b = curr
                new_w, new_b = new
                inter_w = torch.min(curr_w, new_w)
                inter_b = torch.min(curr_b, new_b)

                intersect._param_u[i] = [inter_w, inter_b]

        for task in val:
            inputs, targets = next(iter(DataLoader(task, batch_size=128)))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            print(
                f"{interval_utils._get_min_acc(intersect, inputs, targets).item():.4f}",
                end=" ",
            )
        print()

        return intersect

    def _get_bounds_intersection(
        self, bounds: list[list[IntervalBoundedModel]], certificates: list[list[float]]
    ) -> tuple[list[IntervalBoundedModel], list[float]]:
        intersections = []
        certs = []
        for bound_pairs, certificate_pairs in zip(
            itertools.product(*bounds), itertools.product(*certificates)
        ):
            intersection = self._generate_intersection(bound_pairs)
            intersections.append(intersection)
            certs.append(certificate_pairs)

        return intersections, certs
