from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, ConcatDataset, Dataset

class Buffer(ABC):
    def __init__(self):
        self._utilization_count = 0

    @property
    def utilization(self):
        return self._utilization_count

    def consume_merge(self, dataset=None):
        r = self.consume()
        if r[0] is None:
            return None, None, r[1]
        X, Y = r[0]
        new_dataset = TensorDataset(X, Y)
        if dataset:
            new_dataset = ConcatDataset([dataset, new_dataset])

        return new_dataset, (X, Y)

    @abstractmethod
    def is_empty(self):
        raise NotImplementedError

    def consume(self) -> tuple[tuple[torch.Tensor, torch.Tensor], int | None]:
        """
        Consume elements from the buffer to get fresh verification data
        :return: the fresh data and (None, id) with id of the task in case it has no elements for verification
        """
        self._utilization_count += 1
        return self._consume_buffer()

    @abstractmethod
    def _consume_buffer(self) -> tuple[tuple[torch.Tensor, torch.Tensor], int | None]:
        raise NotImplementedError

    @abstractmethod
    def drain_buffer(self):
        raise NotImplementedError

    @abstractmethod
    def samples_consumed(self):
        raise NotImplementedError

    @abstractmethod
    def add_data(self, data: Dataset) -> None:
        raise NotImplementedError
