import torch
from torch.utils.data import ConcatDataset, Dataset

from src.buffer import Buffer
from src.buffer.SingleTaskBuffer import SingleTaskBuffer


class MultiTaskBuffer(Buffer):
    def __init__(self, buffer_list: list[Buffer]):
        super().__init__()
        self._buffers = buffer_list.copy()

    def _consume_buffer(self) -> tuple[tuple[torch.Tensor, torch.Tensor], int | None]:
        if not self._buffers:
            return (None, None), None
        for b, id in zip(self._buffers, range(len(self._buffers))):
            if b.is_empty():
                return (None, None), id

        X, Y = self._buffers[0].consume()[0]
        for i in range(1, len(self._buffers)):
            Xn, Yn = self._buffers[i].consume()[0]
            if not isinstance(X, torch.Tensor):
                X = (torch.cat([X[0], Xn[0]], dim=0), torch.cat([X[1], Xn[1]], dim=0))
            else:
                X = torch.cat([X, Xn], dim=0)
            Y = torch.cat([Y, Yn], dim=0)

        return (X, Y), None

    def is_empty(self):
        if not self._buffers:
            return True
        for b in self._buffers:
            if b.is_empty():
                return True
        return False

    def drain_buffer(self):
        datasets = []
        for buffer in self._buffers:
            data = buffer.drain_buffer()
            if data:
                datasets.append(data)
        if not datasets:
            return None

        return ConcatDataset(datasets)

    def samples_consumed(self):
        consumed = 0
        for buffer in self._buffers:
            consumed += buffer.samples_consumed()
        return consumed

    def current_tasks_in_buffer(self):
        return [buffer._name for buffer in self._buffers]

    def add_data(self, data: Dataset, task_id: int, k: int = 200) -> None:
        k = (
            self._buffers[0]._k
            if self._buffers
            else k
            if k < len(data)
            else len(data)
        )
        buffer = SingleTaskBuffer(data, k, task_id=task_id)
        self._buffers.append(buffer)

    def drop_buffer(self, id: int) -> None:
        if id < len(self._buffers) - 1:
            self._buffers = self._buffers[:id] + self._buffers[id+1:]
        elif id < len(self._buffers):
            self._buffers = self._buffers[:id]

    def num_of_tasks(self) -> int:
        return len(self._buffers)
