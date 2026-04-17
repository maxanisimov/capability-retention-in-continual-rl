import torch
import torch.nn as nn


class SITracker:
    def __init__(self, model: nn.Module, active: bool = True):
        self.model = model
        self.active = active
        self.importance_scores = [
            torch.zeros_like(p, requires_grad=False) for p in self.model.parameters()
        ]
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]

    def update_importance(self):
        if not self.model.training or not self.active:
            return

        for param, prev_param, omega in zip(
            self.model.parameters(), self.prev_params, self.importance_scores
        ):
            if param.grad is not None:
                # Use .detach() to ensure we're not altering the computation graph
                delta_theta = param.detach() - prev_param
                # The core SI calculation: importance += gradient * (-delta_theta)
                # We use -delta_theta because a move against the gradient indicates importance
                # for reducing the loss. We want a positive importance score.
                omega += param.grad.detach() * (-delta_theta)
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]

    def activate(self) -> None:
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def reset(self):
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]
        self.importance_scores = [
            torch.zeros_like(p, requires_grad=False) for p in self.model.parameters()
        ]
