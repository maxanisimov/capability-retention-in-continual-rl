from src.trainer import IntervalTrainer, AGEMTrainer
from src.regulariser import BaseRegulariser
from src.data_utils import get_batch
import src.utils.general as utils

from typing import Callable
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


class IntervalAGEMTrainer(IntervalTrainer):
    def __init__(
        self,
        model: nn.Module,
        projection_strategy: str = "closest",
        n_certificate_samples: int = 256,
        min_acc_increment: float = 0.05,
        min_acc_limit: float = 0.9,
        memory_samples: int = 100,
        paradigm: str = "TIL",
        domain_map_fn: Callable = None,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(
            model,
            projection_strategy,
            n_certificate_samples,
            min_acc_increment,
            min_acc_limit,
            paradigm,
            domain_map_fn,
            seed,
            **rashomon_kwargs,
        )

        self.memory_samples = memory_samples
        self.memory_batches = []

    def _train_step(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        memory_batches: list[tuple[torch.Tensor, torch.Tensor]] = [],
        regulariser: BaseRegulariser = None,
        project: bool = True,
        context_id: int = None,
        **kwargs: dict,
    ) -> tuple[float, float]:
        model.train()
        inputs, targets = X.to(self.device), y.to(self.device)
        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            targets = self.domain_map_fn(targets)
        self._set_context(model, context_id)

        optimizer.zero_grad()
        outputs_curr = model(inputs)
        loss_curr = loss_fn(outputs_curr, targets)
        if regulariser is not None:
            loss_curr += regulariser(model=model, inputs=inputs, targets=targets)
        loss_curr.backward()
        grad_curr = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.requires_grad]
        ).detach()

        acc = (outputs_curr.argmax(dim=1) == targets).sum() / len(targets)
        if not memory_batches:
            optimizer.step()
            return loss_curr.item(), acc.item()

        inputs_list, targets_list = zip(*memory_batches)
        mem_dataset = TensorDataset(
            torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)
        )
        x_mem, y_mem = get_batch(
            mem_dataset,
            len(mem_dataset),
            shuffle=True,
            seed=self.seed,
            device=self.device,
        )
        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            y_mem = self.domain_map_fn(y_mem)

        optimizer.zero_grad()
        outputs_mem = model(x_mem)
        loss_mem = loss_fn(outputs_mem, y_mem)
        if regulariser is not None:
            loss_mem += regulariser(model=model, inputs=x_mem, targets=y_mem)
        loss_mem.backward()
        grad_mem = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.requires_grad]
        ).detach()

        dot_prod = grad_curr @ grad_mem
        if dot_prod < 0:
            grad_tilde = grad_curr - (dot_prod / (grad_mem @ grad_mem)) * grad_mem
        else:
            grad_tilde = grad_curr

        optimizer.zero_grad(set_to_none=False)
        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                p_len = p.numel()
                p.grad.data = grad_tilde[offset : offset + p_len].view_as(p)
                offset += p_len

        optimizer.step()

        if project:
            self._project_parameters(model, self.bounds, inputs, targets)

        return loss_curr.item(), acc.item()

    def _train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None
        stopping = False

        self._set_context(model, kwargs.get("context_id", None))

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                loss, acc = self._train_step(
                    model,
                    inputs,
                    targets,
                    optimizer,
                    loss_fn,
                    self.memory_batches,
                    regulariser,
                    **kwargs,
                )

                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "acc": f"{acc:.4f}",
                        "val_acc": f"{val_acc:.4f}" if val_acc is not None else None
                    }
                )

                if max(1, i) % kwargs.get("val_freq", 100) == 0:
                    val_loss, val_acc = self._validate_model(
                        model,
                        val_loader,
                        loss_fn,
                        kwargs.get("context_id", None),
                    )

                    pbar.set_postfix(
                        {
                            "loss": f"{loss:.4f}",
                            "acc": f"{acc:.4f}",
                            "val_acc": f"{val_acc:.4f}" if val_acc is not None else None
                        }
                    )

                    if early_stopping:
                        stopping, epochs_no_improve = self.check_early_stopping(
                            curr_loss=val_loss,
                            prev_loss=best_val_loss
                            if best_val_loss != float("inf")
                            else None,
                            patience=kwargs.get("patience", 5),
                            no_improvement=epochs_no_improve,
                        )

                        best_val_loss = min(val_loss, best_val_loss)

                        if stopping:
                            print(f"Early stopping at epoch {epoch + 1}")
                            if best_model_state is not None:
                                model.load_state_dict(best_model_state)
                            break
                        elif val_loss == best_val_loss:
                            best_model_state = model.state_dict()
            if stopping:
                break

        self.memory_batches.append(
            get_batch(
                train_loader.dataset,
                self.memory_samples,
                device="cpu",
                seed=self.seed,
            )
        )

        return model
