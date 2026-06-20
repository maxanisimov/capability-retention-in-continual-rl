from abstract_gradient_training.bounded_models import IntervalBoundedModel, BoundedModel
from src.IntervalTensor import IntervalTensor
import src.verification.verify as verify
from src.rashomon_spec import AccuracyRequirement, RashomonCertificate, RashomonResult

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


def _unpack_batch(
    batch: tuple[torch.Tensor, ...], has_input_intervals: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack a DataLoader batch into (x_l, x_u, y), handling both point datasets
    (batches of (x, y), giving x_l=x_u=x) and interval datasets (batches of (x_l, x_u, y)).
    """
    if has_input_intervals:
        x_l, x_u, y = batch
        return x_l, x_u, y
    x, y = batch
    return x, x, y


def _get_min_acc(
    bounded_model: IntervalBoundedModel,
    X_l: torch.Tensor,
    X_u: torch.Tensor,
    y: torch.Tensor,
    accuracy: AccuracyRequirement,
    soft: bool = False,
    lower: bool = True,
    context_mask: torch.Tensor | None = None,
    multi_label: bool = False,
) -> torch.Tensor:
    """
    Compute the minimum accuracy of the model on the given data using IBP.

    Args:
        bounded_model: The interval-bounded model
        X_l: Lower bound of the input region (equal to X_u for point inputs)
        X_u: Upper bound of the input region (equal to X_l for point inputs)
        y: Target labels. For multi-label case, should be a multi-hot tensor of shape
           (N, n_classes) with 1 for valid actions and 0 otherwise
        accuracy: Specifies the soft surrogate metric, its temperature, and the
            within-group sample aggregation to use
        soft: Whether to use soft accuracy
        lower: Whether to compute lower bound (True) or upper bound (False)
        context_mask: Optional context mask
        multi_label: If True, treats y as containing multiple valid labels per sample

    Returns:
        Accuracy bound tensor
    """
    logits = IntervalTensor(*bounded_model.bound_forward(X_l, X_u))
    if context_mask is not None:
        logits = logits * context_mask

    if multi_label:
        if soft:
            if accuracy.soft_metric == "soft_accuracy":
                acc = verify.bound_multi_label_soft_accuracy(
                    logits, y, T=accuracy.soft_temperature, lower=lower,
                    aggregation=accuracy.aggregation,
                )
            elif accuracy.soft_metric == "accuracy_margin":
                acc = verify.bound_multi_label_accuracy_margin(
                    logits, y, T=accuracy.soft_temperature, lower=lower,
                    aggregation=accuracy.aggregation,
                )
            else:
                raise ValueError(
                    f"Unsupported multi-label soft metric: {accuracy.soft_metric}"
                )
        else:
            # route the hard multi-label "argmax in admissible set" check through the
            # shared verify_point primitive, so this and ad-hoc pointwise verification
            # share one code path.
            from src.verification.api import AdmissibleSet, verify_point

            result = verify_point(
                bounded_model, AdmissibleSet(n_classes=y.shape[-1], multi_hot=y),
                x_l=X_l, x_u=X_u, lower=lower, context_mask=context_mask,
            )
            per_sample = result.certified.float()
            if accuracy.aggregation == "min":
                acc = per_sample.min()
            elif accuracy.aggregation == "mean":
                acc = per_sample.mean()
            else:
                raise ValueError(f"Unsupported aggregation method: {accuracy.aggregation}")
    else:
        if soft:
            acc = verify.bound_soft_accuracy(logits, y, T=accuracy.soft_temperature, lower=lower)
        else:
            acc = verify.bound_accuracy(logits, y, lower=lower)
    return acc


def _certify_groups(
    bounded_model: IntervalBoundedModel,
    X_l: torch.Tensor,
    X_u: torch.Tensor,
    y: torch.Tensor,
    accuracy: AccuracyRequirement,
    groups: list[int | None],
    group_by: Callable[[torch.Tensor], torch.Tensor] | None,
    multi_label: bool,
    context_mask: torch.Tensor | None,
) -> list[RashomonCertificate]:
    """Certify a bounded model against a certificate batch, once per group."""
    group_ids = group_by(y) if group_by is not None else None
    certs = []
    for group in groups:
        mask = (
            torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
            if group is None
            else group_ids == group
        )
        if not mask.any():
            continue
        hard_acc = _get_min_acc(
            bounded_model, X_l[mask], X_u[mask], y[mask], accuracy,
            soft=False, context_mask=context_mask, multi_label=multi_label,
        ).item()
        soft_acc = _get_min_acc(
            bounded_model, X_l[mask], X_u[mask], y[mask], accuracy,
            soft=True, context_mask=context_mask, multi_label=multi_label,
        ).item()
        certs.append(RashomonCertificate(group=group, min_soft_acc=soft_acc, min_hard_acc=hard_acc))
    return certs


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
        accuracy: AccuracyRequirement,
        penalty_coefficient: float,
        objective_fn: Callable,
        obj_alpha: float,
        context_mask: torch.Tensor | None = None,
        domain_map_fn: Callable | None = None,
        group_by: Callable[[torch.Tensor], torch.Tensor] | None = None,
        multi_label: bool = False,
        has_input_intervals: bool = False,
    ):
        super().__init__()
        self.bounded_model = bounded_model
        device = bounded_model.device
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.obj_alpha = obj_alpha
        self.group_by = group_by
        self.has_input_intervals = has_input_intervals

        full_batch = next(iter(torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False,
        )))
        _, _, all_y = _unpack_batch(full_batch, has_input_intervals)
        self.groups: list[int | None] = (
            sorted(group_by(all_y).unique().tolist()) if group_by is not None else [None]
        )

        self.context_mask = context_mask
        self.objective_fn = objective_fn
        self.domain_map_fn = domain_map_fn
        self.multi_label = multi_label
        self.accuracy = accuracy
        self.penalty_updater = (
            cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
                growth_factor=1.01, violation_tolerance=0.05
            )
        )
        self.new = True
        for i in range(len(self.groups)):
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
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        X_l, X_u, y = _unpack_batch(batch, self.has_input_intervals)

        X_l = X_l.to(self.bounded_model.device)
        X_u = X_u.to(self.bounded_model.device)
        y = y.to(self.bounded_model.device)

        # apply projection
        _project_bounded_model(self.bounded_model, context_mask=self.context_mask)
        loss = -self.objective_fn(self.bounded_model, self.obj_alpha)

        misc_info = {}
        observed_constraints = {}

        group_ids = self.group_by(y) if self.group_by is not None else None

        for i, group in enumerate(self.groups):
            mask = (
                torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
                if group is None
                else group_ids == group
            )
            if not mask.any():
                continue
            inputs_l, inputs_u, targets = X_l[mask], X_u[mask], y[mask]

            if self.domain_map_fn is not None:
                targets = self.domain_map_fn(targets)
            soft_min_acc = _get_min_acc(
                self.bounded_model,
                inputs_l,
                inputs_u,
                targets,
                self.accuracy,
                soft=True,
                context_mask=self.context_mask,
                multi_label=self.multi_label,
            )
            min_acc = _get_min_acc(
                self.bounded_model,
                inputs_l,
                inputs_u,
                targets,
                self.accuracy,
                soft=False,
                context_mask=self.context_mask,
                multi_label=self.multi_label,
            )

            soft_limit, hard_limit = self.accuracy.resolve(group)
            defect = soft_limit - soft_min_acc
            setattr(self, f"defect{i}", defect)
            defect_strict = hard_limit - min_acc
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
                "target_min_acc": hard_limit,
                "target_min_soft_acc": soft_limit,
                "penalty": getattr(self, f"constraint{i}").penalty_coefficient().item(),
            }
            observed_constraints = observed_constraints | {
                getattr(self, f"constraint{i}"): constraint_state
            }

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
                [getattr(cmp, f"defect_strict{i}") for i in range(len(cmp.groups))]
            )
        )
        min_violation = torch.min(
            torch.tensor(
                [getattr(cmp, f"defect_strict{i}") for i in range(len(cmp.groups))]
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


def compute_rashomon_set(
    model: torch.nn.Sequential,
    dataset: torch.utils.data.Dataset,
    accuracy: AccuracyRequirement,
    *,
    batch_size: int = 500,
    certificate_samples: int = 1000,
    n_iters: int = 2000,
    primal_learning_rate: float = 0.1,
    dual_learning_rate: float = 0.1,
    penalty_coefficient: float = 1.0,
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
    group_by: Callable[[torch.Tensor], torch.Tensor] | None = None,
    multi_label: bool = False,
    has_input_intervals: bool = False,
    certification_method: str = "IBP",
    certification_method_kwargs: dict | None = None,
) -> RashomonResult:
    """
    Computes the Rashomon set using Lagrangian optimization with the Cooper library.

    Args:
        model (torch.nn.Sequential): The model to optimize.
        dataset (torch.utils.data.Dataset): The dataset for certification. Yields (x, y) batches
            by default, or (x_l, x_u, y) batches if has_input_intervals=True.
        accuracy (AccuracyRequirement): Specifies the soft/hard accuracy limits (optionally
            per-group), the soft surrogate metric and temperature, and the within-group sample
            aggregation.
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
        group_by (Callable, optional): Function applied to each minibatch's `y` (or per-row multi-hot
            admissible-set tensor) producing an integer group-id tensor. Each unique group gets its own
            Lagrangian constraint with its own (soft_min, hard_min) limits, resolved via
            `accuracy.resolve(group)`. If None, all samples form a single global group (group id None).
        has_input_intervals (bool): If True, `dataset` yields (x_l, x_u, y) batches (input-region
            certification) instead of (x, y) point batches.
        certification_method (str): Verification method (see `src.verification.registry`) used to compute
            the *reported* certificates. The optimization loop itself always uses IBP for speed; this only
            affects the checkpoint/final certificates, which are computed by rebuilding a BoundedModel of
            this method from each checkpoint's (param_l, param_u).
        certification_method_kwargs (dict, optional): Extra kwargs forwarded to the certification backend.

    Returns:
        RashomonResult: The optimized bounded models (one per checkpoint, IBP-parameterized) and their
            certificates (one list of per-group RashomonCertificate per checkpoint).
    """
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
    # Get the batch of data we'll use to report our certificates w.r.t., balanced per group if group_by is given
    encountered_groups = 1
    if group_by is not None:
        full_batch = next(iter(torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False,
        )))
        X_l_all, X_u_all, y_all = _unpack_batch(full_batch, has_input_intervals)
        group_ids_all = group_by(y_all)
        cert_groups = []
        for group in sorted(group_ids_all.unique().tolist()):
            mask = group_ids_all == group
            if not mask.any():
                continue
            tensors = (
                (X_l_all[mask][:certificate_samples], X_u_all[mask][:certificate_samples], y_all[mask][:certificate_samples])
                if has_input_intervals
                else (X_l_all[mask][:certificate_samples], y_all[mask][:certificate_samples])
            )
            cert_groups.append(torch.utils.data.TensorDataset(*tensors))
        dataset = torch.utils.data.ConcatDataset(cert_groups)
        encountered_groups = len(cert_groups)

    if context_mask is not None:
        bounded_model.param_l[-1].data -= (1 - context_mask) * MAX_PARAMETER_WIDTH
        bounded_model.param_u[-1].data += (1 - context_mask) * MAX_PARAMETER_WIDTH
        bounded_model.param_l[-2].data -= (
            (1 - context_mask) * MAX_PARAMETER_WIDTH
        ).unsqueeze(1)
        bounded_model.param_u[-2].data += (
            (1 - context_mask) * MAX_PARAMETER_WIDTH
        ).unsqueeze(1)
    dl_cert = torch.utils.data.DataLoader(
        dataset,
        batch_size=certificate_samples * encountered_groups,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    X_l_cert, X_u_cert, y_cert = _unpack_batch(next(iter(dl_cert)), has_input_intervals)
    X_l_cert, X_u_cert, y_cert = X_l_cert.to(device), X_u_cert.to(device), y_cert.to(device)
    og_y_cert = y_cert
    if domain_map_fn is not None:
        y_cert = domain_map_fn(y_cert)

    groups: list[int | None] = (
        sorted(group_by(y_cert).unique().tolist()) if group_by is not None else [None]
    )
    resolved_limits = {g: accuracy.resolve(g) for g in groups}

    if all(s <= 0.0 and h <= 0.0 for s, h in resolved_limits.values()):
        print(
            "Warning: both soft_min and hard_min <= 0.0 for every group; returning without computing Rashomon set."
        )
        if outer_bbox is not None:
            bounded_model = copy.deepcopy(outer_bbox)
        else:
            for pl, p, pu in zip(
                bounded_model.param_l, model.parameters(), bounded_model.param_u
            ):
                pl.data = p - MAX_PARAMETER_WIDTH
                pu.data = p + MAX_PARAMETER_WIDTH
        zero_certs = [RashomonCertificate(group=g, min_soft_acc=0.0, min_hard_acc=0.0) for g in groups]
        return RashomonResult(bounded_models=[bounded_model], certificates=[zero_certs])

    # Check initial constraint satisfaction feasibility
    with torch.no_grad():
        initial_certs = _certify_groups(
            bounded_model, X_l_cert, X_u_cert, y_cert, accuracy, groups, group_by,
            multi_label, context_mask,
        )
    for cert in initial_certs:
        soft_limit, _ = resolved_limits[cert.group]
        defect = soft_limit - cert.min_soft_acc
        if defect != defect:  # NaN check
            raise ValueError("Initial bmodel results in NaN defect for the constraint.")
        print(
            f"Initial acc constraint violation (group={cert.group}): {defect:.4f} (Positive = violated)"
        )

    if outer_bbox is not None:
        print(
            f"Computing Rashomon set within outer box of size: {_bounded_model_width(outer_bbox).item():.2f}"
        )
        outer_certs = _certify_groups(
            outer_bbox, X_l_cert, X_u_cert, y_cert, accuracy, groups, group_by,
            multi_label, context_mask,
        )
        if all(
            cert.min_soft_acc > resolved_limits[cert.group][0]
            and cert.min_hard_acc > resolved_limits[cert.group][1]
            for cert in outer_certs
        ):
            print(
                "Warning: outer bounding box already satisfies both thresholds for every group. "
                "No need to compute Rashomon set."
            )
            return RashomonResult(bounded_models=[outer_bbox], certificates=[outer_certs])

    # Instantiate the Constrained Minimization Problem (CMP)
    cmp = BboxOptimizationCMP(
        bounded_model,
        dataloader=dataloader,
        accuracy=accuracy,
        objective_fn=objective_fn,
        penalty_coefficient=penalty_coefficient,
        obj_alpha=obj_alpha,
        context_mask=context_mask,
        domain_map_fn=domain_map_fn,
        group_by=group_by,
        multi_label=multi_label,
        has_input_intervals=has_input_intervals,
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
    print(f"Computing Rashomon set with limits: {resolved_limits}")
    obj = objective_fn(bounded_model, obj_alpha)
    initial_certs = _certify_groups(
        bounded_model, X_l_cert, X_u_cert, y_cert, accuracy, groups, group_by,
        multi_label, context_mask,
    )
    print(
        "Initial bbox: ",
        f"Obj={obj.item():.2f}, ",
        f"Size={_bounded_model_width(bounded_model):.2f}, ",
        f"Certificates={initial_certs}",
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
    final_certs = _certify_groups(
        bounded_model, X_l_cert, X_u_cert, y_cert, accuracy, groups, group_by,
        multi_label, context_mask,
    )
    print(
        "Final bbox: ",
        f"Obj={obj.item():.2f}, ",
        f"Size={_bounded_model_width(bounded_model):.2f}, ",
        f"Certificates={final_certs}",
    )

    # Remove the hooks created for masking
    for handle in hooks:
        handle.remove()

    # compute the final checkpoint certificates over the entire dataset, using certification_method
    print(f"Computing final certificates over {certificate_samples} samples using {certification_method}")
    print("Num cert samples:", len(og_y_cert))

    from src.verification.api import build_bounded_model

    checkpoint_certs: list[list[RashomonCertificate]] = []
    for m in checkpoint_models:
        cert_model = build_bounded_model(
            model, certification_method,
            param_l=[p.detach().clone() for p in m.param_l],
            param_u=[p.detach().clone() for p in m.param_u],
            **(certification_method_kwargs or {}),
        )
        checkpoint_certs.append(
            _certify_groups(
                cert_model, X_l_cert, X_u_cert, og_y_cert, accuracy, groups, group_by,
                multi_label, context_mask,
            )
        )

    if checkpoint != -1:
        print(
            f"Checkpointed every {checkpoint} iterations for a total of {len(checkpoint_models)} checkpoints"
        )
        checkpoint_sizes = [_bounded_model_width(m) for m in checkpoint_models]
        print(f"Checkpoints sizes: {[f'{c:.2f}' for c in checkpoint_sizes]}")
        print(f"Checkpoint certificates: {checkpoint_certs}")

    print(f"{' Finished Computing Rashomon set ':-^80}")
    return RashomonResult(bounded_models=checkpoint_models, certificates=checkpoint_certs)


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
