from src.trainer.BaseTrainer import BaseTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training.bounded_models import BoundedModel
import src.interval_utils as interval_utils
import src.utils.general as utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable
from tqdm import tqdm

import numpy


class IntervalTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        projection_strategy: str = "closest",
        n_certificate_samples=256,
        min_acc_increment=0.05,
        min_acc_limit=0.9,
        paradigm: str = "TIL",
        domain_map_fn: Callable = None,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(
            model=model, paradigm=paradigm, domain_map_fn=domain_map_fn, seed=seed
        )
        self._last_projection = None
        self.projection_strategy = projection_strategy
        self.n_certificate_samples = n_certificate_samples
        self.min_acc_increment = min_acc_increment
        self.min_acc_limit = min_acc_limit
        self.bounds = []
        self.paradigm = paradigm
        self.rashomon_kwargs = rashomon_kwargs
        self.final_certificates = []
        self.certificates = []

    def get_current_bbox(self) -> BoundedModel | None:
        """Get the current bounding box."""
        if self._last_projection is not None:
            bounded_model = self.bounds[self._last_projection]
            if utils.check_inclusion(self.model, bounded_model):
                raise ValueError(
                    f"Current model parameters are not within the bounds of the bounding box {self._last_projection}."
                )
            return self.bounds[self._last_projection]
        return None

    @torch.no_grad()
    def _project_parameters(
        self,
        model: torch.nn.Module,
        bounds: list[BoundedModel],
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> None:
        if not bounds: # No need to project if bounds are yet to be computed
            return
        
        distances = []
        for i, bounded_model in enumerate(bounds):
            distance = 0
            for p_l, p, p_u in zip(
                bounded_model.param_l, model.parameters(), bounded_model.param_u
            ):
                # compute a distance to the bounds, which should be 0 if all parameters are within the bounds
                distance += torch.square(
                    (p_l - p).clamp(min=0) + (p - p_u).clamp(min=0)
                ).sum()
            distances.append(distance)

        if any(
            d <= 1e-6 for d in distances
        ):  # all parameters are within at least one bbox, no need to project
            # Find any index where d <= 1e-6 and set self._last_projection
            self._last_projection = next(
                i for i, d in enumerate(distances) if d <= 1e-6
            )
            return

        if self.projection_strategy == "closest" or X is None:
            # find the closest bound on project onto it
            self._last_projection = distances.index(min(distances))
        elif self.projection_strategy == "sample_closest":
            # sample from the bounding boxes based on their current distance, closer boxes are more likely to be sampled
            probs = (1.0 / torch.tensor(distances)).softmax(dim=0)
            self._last_projection = torch.multinomial(probs, 1).item()
        elif self.projection_strategy == "sample_largest_closest":
            # sample from the bounding boxes based on their current distance, closer boxes are more likely to be sampled
            sizes = torch.tensor(
                [interval_utils._bounded_model_width(bm) for bm in bounds]
            )
            probs = (sizes / torch.tensor(distances)).softmax(dim=0)
            try:
                self._last_projection = torch.multinomial(probs, 1).item()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error sampling from the bounding boxes.\nSizes: {sizes},\nDistances: {distances},\nProbs: {probs}"
                ) from e
        elif self.projection_strategy == "best_loss":
            # the most expensive but potentially optimal strategy
            old_params = [p.clone() for p in model.parameters()]
            losses = []
            for bounded_model in bounds:
                # project the parameters to the bounds
                for p_l, p, old_p, p_u in zip(
                    bounded_model.param_l,
                    model.parameters(),
                    old_params,
                    bounded_model.param_u,
                ):
                    p.data.copy_(old_p.clamp(min=p_l, max=p_u))
                # compute the loss
                losses.append(torch.nn.functional.cross_entropy(model(X), y).item())
            # restore the parameters
            for p, old_p in zip(model.parameters(), old_params):
                p.data.copy_(old_p)
            # find the best loss and project onto it
            self._last_projection = losses.index(min(losses))
        else:
            raise ValueError(f"Unknown projection strategy: {self.projection_strategy}")

        bounded_model = bounds[self._last_projection]
        for p_l, p, p_u in zip(
            bounded_model.param_l, model.parameters(), bounded_model.param_u
        ):
            # project the parameters to the bounds
            p.data.clamp_(min=p_l, max=p_u)

    def compute_rashomon_set(
        self,
        dataset: torch.utils.data.Dataset,
        callback: Callable | None = None,
        use_outer_bbox: bool = True,
        context_id: int | None = None,
        param_mask: Iterable | None = None,
        multi_label: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        Compute the Rashomon set on the given data.
        """
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=self.n_certificate_samples, shuffle=True
        )
        X, y = next(iter(dl))
        X, y = X.to(self.device), y.to(self.device)

        if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
            y = self.domain_map_fn(y)

        self._set_context(self.model, context_id)

        model_class_preds = self.model(X).argmax(dim=1)
        if not multi_label:
            task_acc = (model_class_preds == y).float().mean()
        else:
            task_acc = float(numpy.mean([pred in y[i] for i, pred in enumerate(model_class_preds)]))
        
        if isinstance(self.model[-1], utils.InContextHead):
            model = self.model[:-1]
            context_mask = self.model[-1].mask
        else:
            model = self.model
            context_mask = None

        if isinstance(self.min_acc_limit, list):
            min_acc_limit = self.min_acc_limit
        elif self.min_acc_increment and self.min_acc_limit:
            min_acc_limit = min(max(task_acc - self.min_acc_increment, task_acc / 2), self.min_acc_limit)
        elif self.min_acc_increment:
            min_acc_limit = max(task_acc - self.min_acc_increment, task_acc / 2)
        elif self.min_acc_limit:
            min_acc_limit = self.min_acc_limit
        else:
            raise ValueError(
                "min_acc_limit or min_acc_increment must be set to compute the Rashomon set."
            )

        bounded_models, certificates = interval_utils.compute_rashomon_set(
            model,
            dataset,
            min_acc_limit=min_acc_limit,
            context_mask=context_mask,
            callback=callback,
            certificate_samples=self.n_certificate_samples,
            domain_map_fn=self.domain_map_fn
            if hasattr(self, "domain_map_fn")
            else None,
            outer_bbox=self.get_current_bbox() if use_outer_bbox else None,
            param_mask=param_mask,
            multi_label=multi_label,
            task_labels=[0],
            **self.rashomon_kwargs,
        )
        self.bounds = bounded_models
        self.certificates = certificates
        # we are now in any of the rashomon sets, but we'll use the last which should be the biggest
        # (but maybe not the best)
        self._last_projection = len(bounded_models) - 1

    def _train_step(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
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
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if regulariser is not None:
            loss += regulariser(model=model, inputs=inputs, targets=targets)
        loss.backward()
        optimizer.step()
        if project:
            self._project_parameters(model, self.bounds, inputs, targets)

        acc = (outputs.argmax(dim=1) == targets).sum() / len(targets)

        return loss, acc

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
    ) -> nn.Module:

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None
        val_loss = 0
        stopping = False

        self._set_context(model, kwargs.get("context_id", None))

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                loss, _ = self._train_step(
                    model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                )

                pbar.set_postfix(
                    {
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.4f}"
                        if val_acc is not None
                        else None,
                        "proj": self._last_projection,
                        "progress": f"{i / len(train_loader):.2f}"
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
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc:.4f}"
                            if val_acc is not None
                            else None,
                            "proj": self._last_projection,
                            "progress": f"{i / len(train_loader):.2f}"
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
        
        if self._last_projection is not None:
            if isinstance(self.certificates[-1], list):
                for i, cert in enumerate(self.certificates[self._last_projection]):
                    if cert is not None and i < len(self.final_certificates):
                        self.final_certificates[i] = cert
                    elif cert is not None:
                        self.final_certificates.append(cert)
            else:
                self.final_certificates.append(self.certificates[self._last_projection])

        return model

    @torch.no_grad()
    def _test(
        self, test_loader: DataLoader, loss_fn: Callable, **kwargs: dict
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0

        for i, (inputs, targets) in enumerate(test_loader):
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
