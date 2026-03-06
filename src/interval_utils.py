from abstract_gradient_training.bounded_models import IntervalBoundedModel, BoundedModel
from src.IntervalTensor import IntervalTensor
import src.verification.verify as verify
from src.data_utils import _extract_targets

import torch
import torch.nn as nn
from typing import Callable, Iterable
import copy
import cooper
import tqdm

MAX_PARAMETER_WIDTH = 10.0
SOFT_ACC_TEMP = 10


def _bounded_model_width(bounded_model: IntervalBoundedModel) -> torch.Tensor:
    """Compute the width of the bounding box in parameter space."""
    width = torch.tensor(0.0, device=bounded_model.device)
    for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
        width += (p_u - p_l).sum()
    return width.sum()


def _objective_fn(
    bounded_model: IntervalBoundedModel, alpha: float, mask: Iterable | None = None
) -> torch.Tensor:
    """
    This is the objective function for the primal problem that the optimizer will try to maximize.

    We use a weighted combination of:

        - the log volume of the bounding box, which overly penalises parameters with small bounds (i.e. dense bounds)
        - the width of each parameter interval, which encourages parameters with large bounds (i.e. sparse bounds)

    Args:
        bounded_model (agt.bounded_models.IntervalBoundedModel): The current bounding box over parameters.
        alpha (float): Weight for the log volume vs width in the objective function. Higher value = more dense.

    Returns:
        torch.Tensor: The objective value.
    """
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
    logvol = torch.tensor(0.0, device=bounded_model.device)
    width = torch.tensor(0.0, device=bounded_model.device)
    nparams = 0
    if mask:
        for p_l, p_u, m in zip(bounded_model.param_l, bounded_model.param_u, mask):
            p_l *= ~m
            p_u *= ~m
            logvol += torch.log(p_u - p_l + 1e-6).sum()
            width += (p_u - p_l).sum()
            nparams += p_l.numel()
    else:
        for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
            logvol += torch.log(p_u - p_l + 1e-6).sum()
            width += (p_u - p_l).sum()
            nparams += p_l.numel()

    objective = (alpha * logvol + (1 - alpha) * width).sum()

    if torch.isnan(objective):
        raise ValueError(
            "Objective function returned NaN. Check the model parameters and bounds."
        )

    return objective / nparams


def _get_min_acc(
    bounded_model: IntervalBoundedModel,
    X: torch.Tensor,
    y: torch.Tensor,
    soft: bool = False,
    lower: bool = True,
    context_mask: torch.Tensor | None = None,
    multi_label: bool = False,
) -> torch.Tensor:
    """
    Compute the minimum accuracy of the model on the given data using IBP.
    
    Args:
        bounded_model: The interval-bounded model
        X: Input data tensor
        y: Target labels. For multi-label case, should be a tensor where each row 
           contains valid class indices (padded with -1 for variable length)
        soft: Whether to use soft accuracy
        lower: Whether to compute lower bound (True) or upper bound (False)
        context_mask: Optional context mask
        multi_label: If True, treats y as containing multiple valid labels per sample
    
    Returns:
        Accuracy bound tensor
    """
    logits = IntervalTensor(*bounded_model.bound_forward(X, X))
    if context_mask is not None:
        logits = logits * context_mask
    
    if multi_label:
        if soft:
            acc = verify.bound_multi_label_lse_margin(logits, y, T=SOFT_ACC_TEMP, lower=lower)
        else:
            acc = verify.bound_multi_label_accuracy(logits, y, lower=lower)
    else:
        if soft:
            acc = verify.bound_soft_accuracy(logits, y, T=SOFT_ACC_TEMP, lower=lower)
        else:
            acc = verify.bound_accuracy(logits, y, lower=lower)
    return acc


@torch.no_grad()
def _project_bounded_model(
    bounded_model: IntervalBoundedModel,
    outer_bbox: IntervalBoundedModel | list[IntervalBoundedModel] | None = None,
    context_mask: torch.Tensor | None = None,
) -> IntervalBoundedModel:
    """Project the bounded model to be valid with respect to its center and the outer bounding box."""
    for pl, pn, pu in zip(
        bounded_model.param_l, bounded_model.param_n, bounded_model.param_u
    ):
        pl.clamp_(min=pn - MAX_PARAMETER_WIDTH, max=pn)
        pu.clamp_(min=pn, max=pn + MAX_PARAMETER_WIDTH)

    if isinstance(outer_bbox, list):
        for bbox in outer_bbox:
            for pl, pu, ol, ou in zip(
                bounded_model.param_l, bounded_model.param_u, bbox.param_l, bbox.param_u
            ):
                pl.data.clamp_(min=ol.data, max=ou.data)
                pu.data.clamp_(min=ol.data, max=ou.data)
    elif outer_bbox is not None:
        for pl, pu, ol, ou in zip(
            bounded_model.param_l,
            bounded_model.param_u,
            outer_bbox.param_l,
            outer_bbox.param_u,
        ):
            pl.data.clamp_(min=ol.data, max=ou.data)
            pu.data.clamp_(min=ol.data, max=ou.data)

    return bounded_model


class BboxOptimizationCMP(cooper.ConstrainedMinimizationProblem):
    """Cooper constrained optimization problem for the Rashomon set."""

    def __init__(
        self,
        bounded_model: IntervalBoundedModel,
        dataloader: torch.utils.data.DataLoader,
        min_acc_limits: list[float] | float,
        penalty_coefficient: float,
        objective_fn: Callable,
        obj_alpha: float,
        context_mask: torch.Tensor | None = None,
        domain_map_fn: Callable | None = None,
        task_labels: list[tuple[int, int]] = None,
        multi_label: bool = False,
    ):
        super().__init__()
        self.bounded_model = bounded_model
        device = bounded_model.device
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.obj_alpha = obj_alpha
        self.min_acc_limits = (
            min_acc_limits + [0.0] * (5 - len(min_acc_limits))
            if isinstance(min_acc_limits, list)
            else [min_acc_limits] * 5
        )  # TODO: *5 implies 5 tasks, so not dynamic
        self.context_mask = context_mask
        self.objective_fn = objective_fn
        self.domain_map_fn = domain_map_fn
        self.multi_label = multi_label
        self.penalty_updater = (
            cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
                growth_factor=1.01, violation_tolerance=0.05
            )
        )
        self.new = True
        self.task_labels = (
            task_labels
            or torch._unique(_extract_targets(dataloader.dataset))[0].tolist()
        )  # if there is only one task then we can extract the task labels as opposed to having to provide them
        for i in range(len(self.task_labels)):
            multiplier = cooper.multipliers.DenseMultiplier(
                init=torch.zeros(1, device=device)
            )
            constraint = cooper.Constraint(
                multiplier=multiplier,
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=cooper.formulations.AugmentedLagrangian,
                penalty_coefficient=cooper.penalty_coefficients.DensePenaltyCoefficient(
                    init=torch.tensor(penalty_coefficient, device=device),
                ),
            )
            setattr(self, f"constraint{i}", constraint)
            setattr(self, f"defect_strict{i}", 0.0)

    def compute_cmp_state(self) -> cooper.CMPState:
        """
        Computes the objective and constraints for the current model state.
        """
        # get data
        try:
            X, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            X, y = next(self.data_iter)

        X, y = X.to(self.bounded_model.device), y.to(self.bounded_model.device)

        curr_context_mask = self.context_mask
        # apply projection
        _project_bounded_model(self.bounded_model, context_mask=self.context_mask)
        loss = -self.objective_fn(self.bounded_model, self.obj_alpha)

        misc_info = {}
        observed_constraints = {}
        encountered_tasks = 0

        # NOTE: Hard-coded for multi-label tasks
        if self.multi_label:
            self.task_labels = [0]

        for i, task in enumerate(self.task_labels):
            if not self.multi_label:
                mask = torch.isin(y, torch.tensor(task).to(y.device))
            else:
                mask = torch.ones(y.shape[0], dtype=torch.bool, device=y.device) # NOTE: I need all data samples
            if not mask.any():
                continue
            encountered_tasks += 1
            inputs = X[mask]
            targets = y[mask]

            if encountered_tasks > 1:
                # Set the context mask properly in TIL
                if self.context_mask is not None:
                    # Create a new mask tensor filled with zeros with the same shape and device.
                    # This is a non-in-place operation, which is safe for autograd.
                    new_mask = torch.zeros_like(self.context_mask)

                    # Use advanced indexing to set the specified task labels to 1.
                    # This is also a non-in-place operation on the original tensor.
                    new_mask[self.task_labels[i]] = 1

                    # Replace the old mask with the new one.
                    self.context_mask = new_mask

            if self.domain_map_fn is not None:
                targets = self.domain_map_fn(targets)
            soft_min_acc = _get_min_acc(
                self.bounded_model,
                inputs,
                targets,
                soft=True,
                context_mask=self.context_mask,
                multi_label=self.multi_label,
            )
            min_acc = _get_min_acc(
                self.bounded_model,
                inputs,
                targets,
                soft=False,
                context_mask=self.context_mask,
                multi_label=self.multi_label,
            )

            defect = self.min_acc_limits[i] - soft_min_acc
            setattr(self, f"defect{i}", defect)
            defect_strict = self.min_acc_limits[i] - min_acc
            setattr(self, f"defect_strict{i}", defect_strict)
            constraint_state = cooper.ConstraintState(
                violation=getattr(self, f"defect{i}"),
                strict_violation=getattr(self, f"defect_strict{i}"),
            )

            misc_info = misc_info | {
                "obj": -loss.item(),
                "defect": getattr(self, f"defect{i}").item(),
                "strict_defect": getattr(self, f"defect_strict{i}").item(),
                "min_acc": min_acc.item(),
                "min_soft_acc": soft_min_acc.item(),
                "penalty": getattr(self, f"constraint{i}").penalty_coefficient().item(),
            }
            observed_constraints = observed_constraints | {
                getattr(self, f"constraint{i}"): constraint_state
            }

        self.context_mask = curr_context_mask

        self.penalty_updater.step(observed_constraints)  # type: ignore
        for constraint in observed_constraints.keys():
            constraint.penalty_coefficient.value.clamp_(max=1e3)

        return cooper.CMPState(
            loss=loss,
            observed_constraints=observed_constraints,
            misc=misc_info,
        )


def get_lr_schedulers(
    primal_optimizer: torch.optim.Optimizer,
    dual_optimizer: torch.optim.Optimizer,
    cmp: cooper.ConstrainedMinimizationProblem,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR]:
    """
    Implements the following learning rate scheduling:

        - When the constraints are violated, the primal learning rate is decreased and the
          dual learning rate is increased - this prevents the failure mode where the optimizer
          is unable to return to the feasible region.
        - When the constraints are satisfied, the primal learning rate is increased and the
          dual learning rate is decreased - this speeds up convergence.
    """

    current_multiplier = 1.0

    def lr_lambda(*args) -> float:
        nonlocal current_multiplier
        max_violation = torch.max(
            torch.tensor(
                [getattr(cmp, f"defect_strict{i}") for i in range(len(cmp.task_labels))]
            )
        )
        min_violation = torch.min(
            torch.tensor(
                [getattr(cmp, f"defect_strict{i}") for i in range(len(cmp.task_labels))]
            )
        )
        if (
            max_violation > 0.05
        ):  # decrease primal lr when constraint is violated with margin
            current_multiplier /= 1.1
        elif min_violation < 0.0:  # increase primal lr when constraint is satisfied
            current_multiplier *= 1.1
        current_multiplier = min(
            max(current_multiplier, 1 / 20), 20
        )  # clamp multiplier
        return current_multiplier

    def dual_lr_lambda(*args) -> float:
        nonlocal current_multiplier
        return 1 / current_multiplier

    primal_scheduler = torch.optim.lr_scheduler.LambdaLR(primal_optimizer, lr_lambda)
    dual_scheduler = torch.optim.lr_scheduler.LambdaLR(dual_optimizer, dual_lr_lambda)

    return primal_scheduler, dual_scheduler


def _create_hook(mask: torch.Tensor):
    final_mask = mask.float()

    def hook_fn(grad: torch.Tensor):
        return grad * (1 - final_mask)

    return hook_fn

def update_context_mask_with_bounded_model(task_labels: list[int], current_mask: torch.Tensor, bounded_model: IntervalBoundedModel) -> tuple[torch.Tensor, IntervalBoundedModel]:
    new_mask = torch.zeros_like(current_mask)
    for label in task_labels:
        new_mask[label] = 1

    return new_mask, bounded_model

def compute_rashomon_set(
    model: torch.nn.Sequential,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 500,
    certificate_samples: int = 1000,
    min_acc_limit=0.85,
    n_iters=2000,
    primal_learning_rate=0.1,
    dual_learning_rate=0.1,
    penalty_coefficient=1.0,
    use_schedule: bool = True,
    init_bbox: float = 0.0,
    obj_alpha: float = 0.0,
    callback: Callable | None = None,
    checkpoint: int = -1,
    context_mask: torch.Tensor | None = None,
    outer_bbox: IntervalBoundedModel | None = None,
    custom_objective: Callable | None = None,
    param_select_fn: Callable | None = None,
    domain_map_fn: Callable | None = None,
    param_mask: Iterable | None = None,
    task_labels: list[tuple[float]] = None,
    multi_label: bool = False,
) -> tuple[list[IntervalBoundedModel], list[float]]:
    """
    Computes the Rashomon set using Lagrangian optimization with the Cooper library.

    Args:
        model (torch.nn.Sequential): The model to optimize.
        dataset (torch.utils.data.Dataset): The dataset for certification.
        batch_size (int): Batch size for the dataset.
        certificate_samples (int): Number of samples to use for the final certificates.
        min_acc_limit (float): Minimum accuracy limit for the constraint.
        n_iters (int): Number of iterations for optimization.
        primal_learning_rate (float): Learning rate for primal variables. A higher value prioritizes expanding the bbox,
            potentially at the expense of violating the accuracy constraint.
        dual_learning_rate (float): Learning rate for dual variables. A higher value prioritizes satisfying the
            accuracy constraint, potentially at the expense of the primal loss.
        penalty_coefficient (float): Additional penalty coefficient for the constraint.
        obj_alpha (float): Weight for the log volume vs width in the objective function. Higher value
            prioritizes the log volume (leading to dense bounds), while lower value prioritizes the width
            (leading to sparsity in the bounds).
        callback (Callable, optional): Callback function to call after optimization. It should take two arguments:
            - losses: list of losses at each iteration
            - defects: list of defects at each iteration
        checkpoint (int): Whether to checkpoint the model every n iterations. If -1, no checkpointing is done.
        context_mask (torch.Tensor, optional): Context mask for the model. If None, no context mask is used.
        outer_bbox (agt.bounded_models.IntervalBoundedModel, optional): Outer bounding box for the Rashomon set, which
            we will project the new rashomon bounds into at each iteration. If None, no outer bounding box is used.
        custom_objective (Callable, optional): Custom objective function to use instead of the default one.
        param_select_fn (Callable, optional): Function to select parameters that we wish to compute rashomon sets over.
            If None, all parameters are used.

    Returns:
        bmodel (agt.bounded_models.IntervalBoundedModel): The optimized bounded model.
            If checkpoint is not -1, returns a list of models at each checkpoint.
        certificates (list[float]): The certificates for each checkpoint.
    """
    min_acc_limits = []
    if isinstance(min_acc_limit, list):
        min_acc_limits = min_acc_limit
        min_acc_limit = min_acc_limit[-1]
    device = next(model.parameters()).device
    objective_fn = custom_objective if custom_objective is not None else _objective_fn

    bounded_model = IntervalBoundedModel(model)
    for pl, pu in zip(bounded_model.param_l, bounded_model.param_u):
        pl.data -= init_bbox
        pu.data += init_bbox
        pl.requires_grad = True
        pu.requires_grad = True

    hooks = []
    if param_mask:
        for pl, pu, m in zip(bounded_model.param_l, bounded_model.param_u, param_mask):
            hooks.append(pl.register_hook(_create_hook(m)))
            hooks.append(pu.register_hook(_create_hook(m)))

    # Create the dataloader for the optimization, and get sample the batch we use for our final certificates
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),  # TODO: don't always use 42 seed
    )
    # Get the batch of data we'll use to report our certificates w.r.t.
    encountered_tasks = 1
    encountered_labels = []
    if task_labels and len(task_labels) > 1:
        X, y = next(
            iter(
                torch.utils.data.DataLoader(
                    dataset, batch_size=len(dataset), shuffle=False
                )
            )
        )
        cert_tasks = []
        for task in task_labels:
            mask = torch.isin(y, torch.tensor(task).to(y.device))
            if not mask.any():
                continue
            inputs = X[mask][:certificate_samples]
            targets = y[mask][:certificate_samples]
            cert_tasks.append(torch.utils.data.TensorDataset(inputs, targets))
            encountered_labels += task
        dataset = torch.utils.data.ConcatDataset(cert_tasks)
        encountered_tasks = len(cert_tasks)

    if context_mask is not None:
        mask = context_mask
        if encountered_labels:
            mask = torch.zeros_like(context_mask)
            for label in encountered_labels:
                mask[label] = 1
        bounded_model.param_l[-1].data -= (1 - mask) * MAX_PARAMETER_WIDTH
        bounded_model.param_u[-1].data += (1 - mask) * MAX_PARAMETER_WIDTH
        bounded_model.param_l[-2].data -= (
            (1 - mask) * MAX_PARAMETER_WIDTH
        ).unsqueeze(1)
        bounded_model.param_u[-2].data += (
            (1 - mask) * MAX_PARAMETER_WIDTH
        ).unsqueeze(1)
    dl_cert = torch.utils.data.DataLoader(
        dataset,
        batch_size=certificate_samples * encountered_tasks,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    X_cert, y_cert = next(iter(dl_cert))
    X_cert, y_cert = X_cert.to(device), y_cert.to(device)
    og_y_cert = y_cert
    if domain_map_fn is not None:
        y_cert = domain_map_fn(y_cert)

    if min_acc_limit <= 0.0:
        print(
            "Warning: min_acc_limit <= 0.0, returning without computing Rashomon set."
        )
        if outer_bbox is not None:
            bounded_model = copy.deepcopy(outer_bbox)
        else:
            for pl, p, pu in zip(
                bounded_model.param_l, model.parameters(), bounded_model.param_u
            ):
                pl.data = p - MAX_PARAMETER_WIDTH
                pu.data = p + MAX_PARAMETER_WIDTH
        return [bounded_model], [0.0]

    # Check initial constraint satisfaction feasibility
    with torch.no_grad():
        initial_defect = min_acc_limit - _get_min_acc(
            bounded_model, X_cert, y_cert, min_acc_limit, context_mask=context_mask,
            multi_label=multi_label
        )
    if torch.isnan(initial_defect):
        raise ValueError("Initial bmodel results in NaN defect for the constraint.")
    print(
        f"Initial acc constraint violation: {initial_defect.item():.4f} (Positive = violated)"
    )

    if outer_bbox is not None:
        print(
            f"Computing Rashomon set within outer box of size: {_bounded_model_width(outer_bbox).item():.2f}"
        )
        outer_min_acc = _get_min_acc(
            outer_bbox, X_cert, y_cert, soft=False, context_mask=context_mask,
            multi_label=multi_label
        )
        if outer_min_acc > min_acc_limit:
            print(
                f"Warning: outer bounding box already satisfies min acc limit of {min_acc_limit:.2f} "
                f"with min acc of {outer_min_acc.item():.2f}. No need to compute Rashomon set."
            )
            return [outer_bbox], [outer_min_acc.item()]

    # Instantiate the Constrained Minimization Problem (CMP)
    cmp = BboxOptimizationCMP(
        bounded_model,
        dataloader=dataloader,
        objective_fn=objective_fn,
        min_acc_limits=min_acc_limits or min_acc_limit,
        penalty_coefficient=penalty_coefficient,
        obj_alpha=obj_alpha,
        context_mask=context_mask,
        domain_map_fn=domain_map_fn,
        task_labels=task_labels,
        multi_label=multi_label,
    )
    n_params = sum(p.numel() for p in bounded_model.param_l)
    print(f"Number of model parameters: {n_params}")

    if param_select_fn is not None:
        primal_vars = param_select_fn(bounded_model)
        nprimal = sum(p.numel() for p in primal_vars)
        print(f"Number of primal variables: {nprimal / 2} ({nprimal})")
    else:
        primal_vars = [*bounded_model.param_l, *bounded_model.param_u]

    # Define Primal Optimizer (for model parameters)
    primal_optimizer = torch.optim.Adam(primal_vars, lr=primal_learning_rate / n_params)

    # Define Dual Optimizer (for Lagrange multipliers - Cooper manages multipliers)
    dual_optimizer = torch.optim.SGD(
        cmp.dual_parameters(), lr=dual_learning_rate, maximize=True
    )

    # Instantiate the Cooper Optimizer
    cooper_optimizer = cooper.optim.AlternatingDualPrimalOptimizer(
        cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
    )

    # --- Optimization Loop ---
    print(f"Computing Rashomon set with min acc limit: {min_acc_limit:.2f}")
    obj = objective_fn(bounded_model, obj_alpha)
    print(
        "Initial bbox: ",
        f"Obj={obj.item():.2f}, ",
        f"Size={_bounded_model_width(bounded_model):.2f}, ",
        f"Min acc hard={_get_min_acc(bounded_model, X_cert, y_cert, soft=False, context_mask=context_mask, multi_label=multi_label):.2f}, ",
        f"Min acc soft={_get_min_acc(bounded_model, X_cert, y_cert, soft=True, context_mask=context_mask, multi_label=multi_label):.2f}",
    )
    primal_lr_scheduler, dual_lr_scheduler = get_lr_schedulers(
        primal_optimizer, dual_optimizer, cmp
    )

    losses = []
    defects = []
    checkpoint_models = []

    for iter_ in (pbar := tqdm.trange(n_iters)):
        if checkpoint != -1 and iter_ % checkpoint == 0 and iter_ > 0:
            checkpoint_models.append(copy.deepcopy(bounded_model))

        roll_out = cooper_optimizer.roll()
        if use_schedule:
            primal_lr_scheduler.step()
            dual_lr_scheduler.step()

        # apply projection
        _project_bounded_model(bounded_model, outer_bbox, context_mask)

        # Logging
        losses.append(roll_out.cmp_state.loss.item())
        defects.append(roll_out.cmp_state.misc["defect"])
        pbar.set_postfix(
            {
                "size": f"{_bounded_model_width(bounded_model).item():.2f}",
                "obj": f"{roll_out.cmp_state.misc['obj']:.3f}",
                "min_soft_acc": f"{roll_out.cmp_state.misc['min_soft_acc']:.3f}",
            }
        )

    checkpoint_models.append(bounded_model)

    if callback is not None:
        callback(losses, defects)

    obj = objective_fn(bounded_model, obj_alpha)

    print(
        "Final bbox: ",
        f"Obj={obj.item():.2f}, ",
        f"Size={_bounded_model_width(bounded_model):.2f}, ",
        f"Min acc hard={_get_min_acc(bounded_model, X_cert, y_cert, soft=False, context_mask=context_mask, multi_label=multi_label):.2f}, ",
        f"Min acc soft={_get_min_acc(bounded_model, X_cert, y_cert, soft=True, context_mask=context_mask, multi_label=multi_label):.2f}",
    )

    # Remove the hooks created for masking
    for handle in hooks:
        handle.remove()

    # compute the final checkpoint certificates over the entire dataset
    print(f"Computing final certificates over {certificate_samples} samples")
    checkpoint_certs = []
    multi_task_certs = []
    print("Num cert samples:", len(og_y_cert))

    # NOTE: hard-coded for multi-label tasks
    if multi_label:
        task_labels = None
    if task_labels:
        for j, m in enumerate(checkpoint_models):
            certs = []
            for i, task in enumerate(task_labels):
                if not j:
                    print("Task labels:", task)
                mask = torch.isin(og_y_cert, torch.tensor(task).to(og_y_cert.device))
                if not mask.any():
                    certs.append(None)
                    continue
                inputs = X_cert[mask]
                targets = y_cert[mask]

                if context_mask is not None:
                    context_mask, bounded_model = update_context_mask_with_bounded_model(task, context_mask, bounded_model)

                cert = _get_min_acc(
                    m, inputs, targets, soft=False, context_mask=context_mask, multi_label=multi_label
                ).item()
                certs.append(cert)
            multi_task_certs.append(certs)

    for m in checkpoint_models:
        checkpoint_certs.append(
            _get_min_acc(
                m, X_cert, y_cert, soft=False, context_mask=context_mask, multi_label=multi_label
            ).item()
        )

    if checkpoint != -1:
        print(
            f"Checkpointed every {checkpoint} iterations for a total of {len(checkpoint_models)} checkpoints"
        )
        checkpoint_sizes = [_bounded_model_width(m) for m in checkpoint_models]
        print(f"Checkpoints sizes: {[f'{c:.2f}' for c in checkpoint_sizes]}")
        print(f"Checkpoint certificates: {[f'{c:.2f}' for c in checkpoint_certs]}")
        print(
            f"Multitask certificates: {[f'[{[round(c, 2) if c is not None else None for c in cs]}]' for cs in multi_task_certs]}"
        )

    print(f"{' Finished Computing Rashomon set ':-^80}")
    return checkpoint_models, multi_task_certs if task_labels else checkpoint_certs


def create_violation_mask(model: nn.Module, bounded_model: BoundedModel):
    mask = []
    for p_l, p, p_u in zip(
        bounded_model.param_l, model.parameters(), bounded_model.param_u
    ):
        # compute a distance to the bounds, which should be 0 if all parameters are within the bounds
        layer_mask = ((p_l - p).clamp(min=0) + (p - p_u).clamp(min=0)) == 0
        mask.append(layer_mask)

    return mask


def max_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = torch.nn.functional.one_hot(
        y_true, num_classes=y_pred.shape[1]
    ).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
        torch.ones_like(y_true_one_hot) - y_true_one_hot
    ) * pred_u  # get upper bound prediction for non targets
    return torch.nn.functional.cross_entropy(min_log, y_true)


def min_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = torch.nn.functional.one_hot(
        y_true, num_classes=y_pred.shape[1]
    ).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
        torch.ones_like(y_true_one_hot) - y_true_one_hot
    ) * pred_u  # get upper bound prediction for non targets
    return torch.sum(torch.argmax(min_log, dim=1) == y_true) / y_pred.shape[0]


def get_balanced_min_acc(
    bounded_model: IntervalBoundedModel,
    X: torch.Tensor,
    y: torch.Tensor,
    lower: bool = True,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the balanced minimum accuracy (average of recall for each class)
    of the model on the given data using IBP."""
    logits = IntervalTensor(*bounded_model.bound_forward(X, X))
    if context_mask is not None:
        logits = logits * context_mask

    acc = verify.bound_balanced_accuracy(logits, y, lower=lower)
    return acc
