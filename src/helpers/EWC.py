import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


class EWC:
    def __init__(self, model: nn.Module, criterion: Callable):
        self.model = model
        self.criterion = criterion
        self.optimal_params = None
        self.fisher_matrix = None
        self.device = next(self.model.parameters()).device

    def _compute_fisher_diagonal(self, dataloader: DataLoader) -> None:
        fisher = [
            torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad
        ]
        self.model.eval()
        num_samples = 0

        for inputs, targets in tqdm(dataloader, desc="Computing Fisher Information"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            num_samples += inputs.size(0)
            outputs = self.model(inputs)

            log_likelihood = F.log_softmax(outputs, dim=1)
            log_likelihood_for_targets = log_likelihood[range(len(targets)), targets]

            for ll in log_likelihood_for_targets:
                self.model.zero_grad()
                ll.backward(retain_graph=True)
                for i, p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        fisher[i] += p.grad.data.pow(2)

        self.fisher_matrix = [f / num_samples for f in fisher]
        self.model.train()

    def register_task(self, dataloader: DataLoader) -> None:
        self.optimal_params = [
            p.clone().detach() for p in self.model.parameters() if p.requires_grad
        ]
        self._compute_fisher_diagonal(dataloader)

    def penalty(self) -> torch.Tensor:
        if self.fisher_matrix is None or self.optimal_params is None:
            return torch.tensor(0.0)

        penalty = 0.0
        current_params = [p for p in self.model.parameters() if p.requires_grad]

        for i, p_curr in enumerate(current_params):
            p_opt = self.optimal_params[i]
            f = self.fisher_matrix[i]
            penalty += (f * (p_curr - p_opt).pow(2)).sum()

        return penalty
