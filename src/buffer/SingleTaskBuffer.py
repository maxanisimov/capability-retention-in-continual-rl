import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.buffer import Buffer
from src.data_utils import TaskAugmentedDataset


class SingleTaskBuffer(Buffer):
    def __init__(self, dataset: Dataset, k: int, task_id: int = 0):
        super().__init__()
        self._dataloader = DataLoader(dataset=dataset, batch_size=k, shuffle=True)
        self._data_it = iter(self._dataloader)
        self._dataset = dataset
        self._consumed = 0
        self._elements = len(self._dataset)
        self._k = k
        self._task_id = task_id

    def is_empty(self):
        return self._consumed + self._k > self._elements

    def _consume_buffer(self):
        if self.is_empty():
            return (None, None), 0
        else:
            self._consumed += self._k
            return next(self._data_it), None

    def drain_buffer(self):
        if self.is_empty():
            return None
        X, Y = None, None
        while not self.is_empty():
            Xn, Yn = self.consume()[0]
            if X is None and Y is None:
                X, Y = Xn, Yn
            else:
                if not isinstance(X, torch.Tensor):
                    X = (
                        torch.cat([X[0], Xn[0]], dim=0),
                        torch.cat([X[1], Xn[1]], dim=0),
                    )
                else:
                    X = torch.cat([X, Xn], dim=0)
                Y = torch.cat([Y, Yn], dim=0)

        return (
            TensorDataset(X, Y)
            if isinstance(X, torch.Tensor)
            else TaskAugmentedDataset(TensorDataset(X[0], Y), torch.unique(X[1]).item())
        )

    def samples_consumed(self):
        return self._consumed

    def add_data(self, data: Dataset) -> None:
        pass
