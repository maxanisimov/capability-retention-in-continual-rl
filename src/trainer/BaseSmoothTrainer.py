from src.trainer import BaseTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.utils.general import acc

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Callable
from abc import abstractmethod


class BaseSmoothTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        n_certificate_samples: int = 256,
        smooth_limit: float = 0.9,
        domain_map_fn: Callable = None,
        seed: int = 42,
        paradigm: str = "TIL",
        **kwargs: dict,
    ):
        super().__init__(
            model=model, seed=seed, paradigm=paradigm, domain_map_fn=domain_map_fn
        )
        self.n_certificate_samples = n_certificate_samples

        # No bounds instead we define the following:
        self.mu = []
        self.sigma = []
        self.alpha = None

        # Default smoothing parameters:
        self.smooth_metric = kwargs.get("smooth_metric", acc)
        self.smooth_bound = smooth_limit
        self.smooth_steps = kwargs.get("smooth_steps", 10)
        self.smooth_iterations = kwargs.get("smooth_iterations", 100)
        self.smooth_delta = kwargs.get("smooth_delta", 0.05)
        self.smooth_infer_samps = kwargs.get("smoother_iter_samps", 35)
        self.smooth_cheat = 1  # This introduces unsoundness. It speeds things up, but must be 1 for all final runs

        # alpha values to scale sigma by
        self.inflates = [1, 2, 5, 7.5, 10, 15, 25, 35, 50, 65]

    def _is_safe(self, parameters: List[torch.Tensor]):
        """Check if parameters are within the Mahalanobis ball defined by loc, scale, radius."""
        total = 0.0
        for idx, (param, loc, sigma) in enumerate(zip(parameters, self.mu, self.sigma)):
            param = param.detach()
            loc = loc.detach()
            sigma = sigma.to(param.device)

            if torch.isnan(param).any():
                raise ValueError(f"NaNs in parameter at index {idx}")
            if torch.isnan(loc).any():
                raise ValueError(f"NaNs in loc at index {idx}")
            if torch.isnan(sigma).any():
                raise ValueError(f"NaNs in scale at index {idx}")

            mask = sigma != 0  # Masking so that we can avoid divsion by 0 issues
            param_masked = param[mask]
            loc_masked = loc[mask]
            sigma_masked = sigma[mask]

            diff = (param_masked - loc_masked) / sigma_masked
            squared_norm = (diff**2).sum().item()
            total += squared_norm

        return True if total <= (self.alpha**2) + 1e-3 else False

    # Only one projection strategy for now
    @torch.no_grad()
    def _project_parameters(
        self,
        model: torch.nn.Module,
    ) -> None:
        if self._is_safe(model.parameters()):
            return

        total = 0.0
        diffs = []
        current_params = model.parameters()
        for param, loc, sigma in zip(current_params, self.mu, self.sigma):
            sigma = sigma.to(param.device)
            mask = sigma != 0
            diff = (param - loc) / sigma
            diffs.append((diff, mask))
            total += ((diff[mask]) ** 2).sum().item()

        scale = (
            (self.alpha) / (total**0.5) * (1 - 1e-3)
        )  # get the proportional violation of the Mahalanobis ball
        with torch.no_grad():
            for p, loc, sigma, (diff, mask) in zip(
                model.parameters(), self.mu, self.sigma, diffs
            ):
                projected = p.clone()
                projected[mask] = (
                    loc[mask] + scale * sigma[mask] * diff[mask]
                )  # move each of the parameters by the scale of the overall violation
                p.copy_(projected)

        assert self._is_safe(model.parameters()), (
            "model parameters are not within the safe area, despite projection"
        )

    @abstractmethod
    def init_density(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        metric: Callable,
        bound: float,
        iterations: int,
        delta: float,
        **kwargs: dict,
    ) -> float:
        """
        Initializes self.loc and self.scale from interval bounds and certifies the smoothing radius.
        """
        raise NotImplementedError()

    def compute_lid(
        self,
        dataset: torch.utils.data.Dataset,
        context_id: int | None = None,
        **kwargs: dict,
    ) -> None:
        """
        Compute the Rashomon set on the given data.
        """
        self._set_context(self.model, context_id)
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=kwargs.get("batch_size", 10_000),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(self.seed),
                )
            )
        )
        self.init_density(
            batch,
            self.smooth_metric,
            self.smooth_bound,
            self.smooth_iterations,
            self.smooth_delta,
            **kwargs,
        )

    def _train_step(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        regulariser: BaseRegulariser = None,
        context_id: int = None,
        **kwargs: dict,
    ) -> tuple[float, float]:
        model.train()
        inputs, targets = X.to(self.device), y.to(self.device)

        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            targets = self.domain_map_fn(targets)
        self._set_context(model, context_id)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if regulariser is not None:
            loss += regulariser(model=model, inputs=inputs, targets=targets)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == targets).sum() / len(targets)

        return loss, acc

    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int = 5,
        early_stopping: bool = False,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ) -> nn.Module:
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        stopping = False
        val_loss, val_acc = 0, 0

        pbar = tqdm(range(epochs), desc="Training Epochs")
        for epoch in pbar:
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                loss, _ = self._train_step(
                    model,
                    inputs,
                    targets,
                    optimizer,
                    loss_fn,
                    regulariser,
                    context_id=kwargs.get("context_id", None),
                )

                if (
                    self.mu and self.sigma
                ):  # Make sure that a bound has been computed prior to trying any sort of projection
                    self._project_parameters(model)

                pbar.set_postfix(
                    {
                        "train_loss": f"{loss.item():.4f}",
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "progress": f"{((i / len(train_loader)) * 100):.2f}%",
                    }
                )

                if max(1, i) % kwargs.get("val_freq", 100) == 0:
                    val_loss, val_acc = self._validate_model(
                        model,
                        val_loader,
                        loss_fn,
                        context_id=kwargs.get("context_id", None),
                    )

                    pbar.set_postfix(
                        {
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "train_loss": f"{loss.item():.4f}",
                            "progress": f"{i / len(train_loader):.2f}",
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

        return model

    @torch.no_grad()
    def _test(
        self, test_loader: DataLoader, loss_fn: Callable, **kwargs: dict
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if hasattr(self, "domain_map_fn") and self.domain_map_fn:
                targets = self.domain_map_fn(targets)

            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        return round(avg_loss, 4), round(accuracy, 4)
