from src.trainer.BaseTrainer import BaseTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training import interval_arithmetic
from src.interval_utils import max_loss, min_acc

from typing import Callable
import torch.nn as nn
import copy
import torch
from torch.utils.data import DataLoader
import tqdm


class InterContiNetTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        min_acc_increment: float = 0.05,
        min_acc_limit: float = 0.9,
        paradigm: str = "TIL",
        domain_map_fn: Callable = None,
        seed: int = 42,
        **rashomon_kwargs: dict,
    ):
        super().__init__(
            model=model, paradigm=paradigm, domain_map_fn=domain_map_fn, seed=seed
        )

        self.min_acc_limit = min_acc_limit
        self.min_acc_increment = min_acc_increment
        self.rashomon_kwargs = rashomon_kwargs
        self.epsilons = list(
                zip(
                    [
                        nn.Parameter(torch.full_like(m.weight, 1)).to(
                            self.device
                        )
                        for m in model
                        if self.compatible_layer(m)
                    ],
                    [
                        nn.Parameter(torch.full_like(m.bias, 1)).to(
                            self.device
                        )
                        for m in model
                        if self.compatible_layer(m)
                    ],
                )
            )

        self.final_certificates = []

    def compatible_layer(self, layer) -> bool:
        return isinstance(layer, (nn.Linear, nn.Conv2d))

    def compute_rashomon_set(
        self,
        dataset: torch.utils.data.Dataset,
        context_id: int | None = None,
        **kwargs: dict,
    ):
        self._set_context(self.model, context_id)
        if not self.min_acc_limit:
            print("Target acc == 0, no need to recompute LID.")
            self.final_certificates.append(0)
            return
        
        dual = DualModel(
            self.model,
            [
                (
                    list(self.model.parameters())[i].clone().detach(),
                    list(self.model.parameters())[i + 1].clone().detach(),
                )
                for i in range(0, len(list(self.model.parameters())), 2)
            ],
            epsilons=self.epsilons,
            default_epsilon=self.rashomon_kwargs.get("default_eps", 1),
            seed=self.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=kwargs.get("batch_size", 128),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        criterion = max_loss
        optimizer = torch.optim.Adam(
            [w for w, _ in dual.vs] + [b for _, b in dual.vs], lr=kwargs.get("lr", 0.01)
        )

        eps_size = sum([(torch.sum(w) + torch.sum(b)).item() for w, b in dual.epsilons])
        print(f"LID size: {eps_size:.4f}.")

        end = False
        for _ in (pbar := tqdm.trange(kwargs.get("epochs", 500))):
            for x, y in loader:
                x, y = x.to(dual.device), y.to(dual.device)
                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    y = self.domain_map_fn(y)
                optimizer.zero_grad()
                out = dual(x)
                loss = criterion(out, y)
                acc = min_acc(y, out)
                if acc >= self.min_acc_limit:
                    end = True
                    break
                loss.backward()
                # print("eps gradient sum:", max([torch.max(p.grad).item() for p in dual.parameters() if p.grad is not None]))
                optimizer.step()

            eps_size = sum([(torch.sum(w) + torch.sum(b)).item() for w, b in dual.extract_epsilons()])
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.4}", "size": f"{eps_size:.2f}"}
            )
            if end:
                break
        
        self.epsilons = dual.extract_epsilons()
        self.final_certificates.append(acc.item())

        eps_size = sum([(torch.sum(w) + torch.sum(b)).item() for w, b in self.epsilons])
        print(f"LID size: {eps_size:.4f} with certificate of {acc}.")

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

        acc = (outputs.argmax(dim=1) == targets).sum().item() / len(targets)

        return loss.item(), acc

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
        val_acc = None
        stopping = False

        self._set_context(model, kwargs.get("context_id", None))

        wrapped = ModelWrapper(
            model, [w for w, _ in self.epsilons], [b for _, b in self.epsilons]
        )
        optimizer = torch.optim.Adam(
            wrapped.parameters(),
            lr=kwargs.get("lr", 0.001),
            weight_decay=kwargs.get("weight_decay", 0),
        )
        for epoch in (pbar := tqdm.trange(epochs, desc="Training Epochs")):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                loss, _ = self._train_step(
                    wrapped, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                )

                if not i % kwargs.get("val_freq", 100):
                    val_loss, val_acc = self._validate_model(
                        wrapped,
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
                            break
            if stopping:
                break

        return wrapped.result()

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
    
class DualModel(nn.Module):
    def __init__(self, model, prev_params, epsilons=None, default_epsilon=0.1, seed=42):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = copy.deepcopy(model)
        self.prev_params = prev_params
        self.vs = list(
            zip(
                [
                    nn.Parameter(torch.full_like(m.weight, 5)).to(self.device)
                    for m in model
                    if self.valid_layer(m)
                ],
                [
                    nn.Parameter(torch.full_like(m.bias, 5)).to(self.device)
                    for m in model
                    if self.valid_layer(m)
                ],
            )
        )
        if not epsilons:
            self.epsilons = list(
                zip(
                    [
                        nn.Parameter(torch.full_like(m.weight, default_epsilon)).to(
                            self.device
                        )
                        for m in model
                        if self.valid_layer(m)
                    ],
                    [
                        nn.Parameter(torch.full_like(m.bias, default_epsilon)).to(
                            self.device
                        )
                        for m in model
                        if self.valid_layer(m)
                    ],
                )
            )
        else:
            self.epsilons = [
                [w.clone().detach(), b.clone().detach()] for w, b in epsilons
            ]

    def valid_layer(self, layer):
        return isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)

    def extract_epsilons(self):
        w_epsilons, b_epsilons = [], []
        for (w_eps_ast, b_eps_ast), (w_ast, b_ast), layer, (w_v, b_v) in zip(
            self.epsilons,
            self.prev_params,
            [layer for layer in self.model if self.valid_layer(layer)],
            self.vs,
        ):
            w_middle, b_middle = layer.weight, layer.bias
            w_inter = torch.min(
                (w_ast + w_eps_ast) - w_middle, w_middle - (w_ast - w_eps_ast)
            )
            w_eps = torch.nn.functional.sigmoid(w_v) * w_inter

            b_inter = torch.min(
                (b_ast + b_eps_ast) - b_middle, b_middle - (b_ast - b_eps_ast)
            )
            b_eps = torch.nn.functional.sigmoid(b_v) * b_inter

            w_epsilons.append(w_eps)
            b_epsilons.append(b_eps)

        return list(zip(w_epsilons, b_epsilons))

    def forward(self, x):
        i, j = 0, 0
        upper = x
        lower = x
        while i < len(self.model):
            og = self.model[i]
            i += 1
            if not self.valid_layer(og):
                upper = og(upper)
                lower = og(lower)
                continue
            w_eps_ast, b_eps_ast = self.epsilons[j]
            w_ast, b_ast = self.prev_params[j]
            w_middle, b_middle = og.weight, og.bias
            w_v, b_v = self.vs[j]

            w_inter = torch.min(
                (w_ast + w_eps_ast) - w_middle, w_middle - (w_ast - w_eps_ast)
            )
            w_eps = torch.nn.functional.sigmoid(w_v) * w_inter
            b_inter = torch.min(
                (b_ast + b_eps_ast) - b_middle, b_middle - (b_ast - b_eps_ast)
            )
            b_eps = torch.nn.functional.sigmoid(b_v) * b_inter
            j += 1

            W_l, W_u = w_middle - w_eps, w_middle + w_eps
            b_l, b_u = b_middle - b_eps, b_middle + b_eps
            if isinstance(og, torch.nn.Linear):
                lower, upper = interval_arithmetic.propagate_matmul(
                    lower, upper, W_l.T, W_u.T
                )
                lower, upper = lower + b_l, upper + b_u
            elif isinstance(og, torch.nn.Conv2d):
                lower, upper = interval_arithmetic.propagate_conv2d(
                    lower,
                    upper,
                    W_l,
                    W_u,
                    b_l,
                    b_u,
                    stride=og.stride,
                    padding=og.padding,
                )

        return torch.stack((lower, upper), dim=2)


class ModelWrapper(nn.Module):
    def __init__(self, model, epsilon_weights=None, epsilon_bias=None):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = copy.deepcopy(model)
        self.weight_u = [
            nn.Parameter(torch.randn_like(m.weight)).to(self.device)
            for m in model
            if self.valid_layer(m)
        ]
        self.bias_u = [
            nn.Parameter(torch.randn_like(m.bias)).to(self.device)
            for m in model
            if self.valid_layer(m)
        ]
        if not epsilon_weights:
            epsilon_weights = [
                nn.Parameter(torch.ones_like(m.weight)).to(self.device)
                for m in model
                if self.valid_layer(m)
            ]
        if not epsilon_bias:
            epsilon_bias = [
                nn.Parameter(torch.ones_like(m.bias)).to(self.device)
                for m in model
                if self.valid_layer(m)
            ]

        self.epsilon_weight = [
            nn.Parameter(param.clone().detach()) for param in epsilon_weights
        ]
        self.epsilon_bias = [
            nn.Parameter(param.clone().detach()) for param in epsilon_bias
        ]

    def __getitem__(self, key):
        return self.model[key]

    def __setitem__(self, key, item):
        self.model[key] = item

    def valid_layer(self, layer):
        return isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)

    def forward(self, x):
        i, j = 0, 0
        while i < len(self.model):
            og = self.model[i]
            i += 1
            if not self.valid_layer(og):
                x = og(x)
                continue
            combined_weight = (
                og.weight
                + torch.nn.functional.tanh(self.weight_u[j]) * self.epsilon_weight[j]
            )
            combined_bias = (
                og.bias
                + torch.nn.functional.tanh(self.bias_u[j]) * self.epsilon_bias[j]
            )
            j += 1

            if isinstance(og, torch.nn.Linear):
                x = torch.nn.functional.linear(x, combined_weight, combined_bias)
            elif isinstance(og, torch.nn.Conv2d):
                x = torch.nn.functional.conv2d(
                    x,
                    combined_weight,
                    combined_bias,
                    stride=og.stride,
                    padding=og.padding,
                )
        return x

    def extract_weights(self):
        out = []
        for weight_u, bias_u, epsilon_weight, epsilon_bias, layer in zip(
            self.weight_u,
            self.bias_u,
            self.epsilon_weight,
            self.epsilon_bias,
            [layer for layer in self.model if self.valid_layer(layer)],
        ):
            combined_weight = (
                layer.weight + torch.nn.functional.tanh(weight_u) * epsilon_weight
            )
            combined_bias = layer.bias + torch.nn.functional.tanh(bias_u) * epsilon_bias
            out.append((combined_weight, combined_bias))

        return out

    def result(self):
        params = self.extract_weights()
        i = 0
        for layer in self.model:
            if self.valid_layer(layer):
                weights, bias = params[i]
                layer.weight = nn.Parameter(weights)
                layer.bias = nn.Parameter(bias)
                i += 1

        return copy.deepcopy(self.model)

    def parameters(self):
        params = [val for val in self.weight_u + self.bias_u]
        return params
