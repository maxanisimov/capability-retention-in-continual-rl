from src.trainer import SimpleTrainer
from src.regulariser.BaseRegulariser import BaseRegulariser
from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src import utils

import copy
import peft
from tqdm import tqdm, trange
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call
from typing import List, Callable

def _acc(batch: DataLoader, model: torch.nn.Module, domain_map_fn = None) -> float:
    model.eval()  # set model to eval mode
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradients for evaluation
        for x, y in batch:
            x = x.to(next(model.parameters()).device)  # move to same device as model
            y = y.to(x.device)

            if domain_map_fn is not None:
                y = domain_map_fn(y)
                        
            outputs = model(x)
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # multi-class classification
                preds = outputs.argmax(dim=1)
            else:
                # binary classification
                preds = (outputs > 0.5).long().squeeze()
            
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total 

class SmoothTrainer(SimpleTrainer):
    def __init__(
        self,
        model: nn.Module,
        projection_strategy: str = "closest",
        n_certificate_samples=256,
        min_acc_increment=0.05,
        min_acc_limit=0.9,
        domain_map_fn: Callable = None,
        seed: int = 42,
        paradigm: str = "TIL",
    ):
        super().__init__(model=model, seed=seed, paradigm=paradigm)
        self.projection_strategy = projection_strategy
        self.n_certificate_samples = n_certificate_samples
        self.min_acc_increment = min_acc_increment
        self.min_acc_limit = min_acc_limit

        # No bounds instead we define the following:
        self.mu = [] # ! mu
        self.sigma = [] # ! sigma
        self.alpha = None # ! I think this might be the alpha value
        self.num_classes = 10

        # Default smoothing parameters:
        self.smooth_metric = _acc
        self.smooth_bound = min_acc_limit
        self.smooth_steps = 10
        self.smooth_iterations = 100
        self.smooth_delta = 0.05
        self.smooth_infer_samps = 35
        self.smooth_cheat = 1 # This introduces unsoundness. It speeds things up, but must be 1 for all final runs

        # ! make an extension to SmoothTrainer called LoRATrainer or something like that
        # For use with LoRA
        self.peft_model = None  
        self.LoRA_loc = None
        self.LoRA_scale = None
        self.LoRA_scalar = None
        
        # Strategies for density computation
        self.density_strategy = "interval_init"
        self.pre_computed_bound = None
        
    def _is_safe(self, parameters: List[torch.Tensor]):
        """Check if parameters are within the Mahalanobis ball defined by loc, scale, radius.""" # Probability sphere
        total = 0.0
        for idx, (param, loc, sigma) in enumerate(zip(parameters, self.loc, self.scale)):
            param = param.detach()
            loc = loc.detach()
            sigma = sigma.to(param.device)

            if torch.isnan(param).any():
                raise ValueError(f"NaNs in parameter at index {idx}")
            if torch.isnan(loc).any():
                raise ValueError(f"NaNs in loc at index {idx}")
            if torch.isnan(sigma).any():
                raise ValueError(f"NaNs in scale at index {idx}")

            mask = sigma != 0
            param_masked = param[mask]
            loc_masked = loc[mask]
            sigma_masked = sigma[mask]

            diff = (param_masked - loc_masked) / sigma_masked
            squared_norm = (diff ** 2).sum().item()
            total += squared_norm
        #print("Model safe: ", total <= self.radius ** 2, " values: ", total, self.radius ** 2)
        return True if total <= (self.radius ** 2) + 1e-3 else False
        
    # Only one projection strategy for now    
    @torch.no_grad()
    def _project_parameters(
        self,
        model: torch.nn.Module,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> None:
        # ! This LoRA stuff should be in its own trainer module
        if(self.peft_model is not None):
            deltas = []
        
            # Step 1: Collect deltas across modules
            # ! all of the lora parameters across layers
            for module in self.peft_model.modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    A, B = module.lora_A.default.weight, module.lora_B.default.weight
                    A0, B0 = module.A0, module.B0
                    deltas.append((A, A0))
                    deltas.append((B, B0))
        
            # Step 2: Compute total norm
            total_sq_norm = sum(((p - p0) ** 2).sum() for (p, p0) in deltas)
            total_norm = total_sq_norm.sqrt()

            # Step 3: Project if needed
            if total_norm > self.radius: # ? How does this projection work? How can the total norm be used to determine wheter params are out of bounds?
                scale = self.radius / total_norm
                for (p, p0) in deltas:
                    with torch.no_grad():
                        delta = (p - p0).detach()
                        p.data = p0 + scale * delta
            return

        # ! No need to project, model is already within bounds
        if(self._is_safe(model.parameters())):
            return
        
        total = 0.0
        diffs = []
        current_params = self.model.parameters()
        for param, loc, sigma in zip(current_params, self.loc, self.scale):
            sigma = sigma.to(param.device)
            mask = sigma != 0 # ? I dont quite understand this masking?
            diff = (param - loc) / sigma # ! Mahalanobis distance
            diffs.append((diff, mask))
            total += ((diff[mask]) ** 2).sum().item()

        scale = (self.radius) / (total**0.5) * (1 - 1e-3) # numerical stability can be tough
        self._last_projection = -1
        with torch.no_grad():
            for p, loc, sigma, (diff, mask) in zip(self.model.parameters(), self.loc, self.scale, diffs):
                projected = p.clone()
                projected[mask] = loc[mask] + scale * sigma[mask] * diff[mask] # ? Doesnt this project all of the parameters as opposed to just the ones out of bounds?
                p.copy_(projected)
    
        assert self._is_safe(model.parameters()), "model parameters are not within the safe area, despite projection"
        

    def initialize_density_from_LoRA(        
        self,
        batch: DataLoader,
        metric: Callable,
        bound: float,
        steps: int,
        iterations: int,
        delta: float,
    ) -> float:
        # Initialize LoRA-based loc/scale
        self.LoRA_loc = []
        self.LoRA_scale = []
        lora_modules = []
    
        for module in self.peft_model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_modules.append((module, module.lora_A.default.weight, module.lora_B.default.weight))
                self.LoRA_loc.append(module.lora_A.default.weight.detach().clone()) 
                self.LoRA_loc.append(module.lora_B.default.weight.detach().clone())

                # ! 0.01 default sigma for LoRA
                self.LoRA_scale.append(0.01 * torch.ones_like(module.lora_A.default.weight))
                self.LoRA_scale.append(0.01 * torch.ones_like(module.lora_B.default.weight))
    
        inflates = [5, 10, 15, 25, 35, 50, 65]
        max_rad = -1e6
    
        for inflate in inflates: # ! Find the best alpha to obtain the largest density function without violating required accuracy constraints
            success_count = 0

            for _ in trange(iterations): # ! inflate sigma by given factor and then check model accuracy
                new_weights = []
                for loc, scale in zip(self.LoRA_loc, self.LoRA_scale):
                    scale_tensor = scale * inflate
                    noise = torch.normal(mean=0.0, std=scale_tensor).to(loc.device)
                    new_weights.append(loc + noise)
    
                # Apply perturbed weights to LoRA modules
                with torch.no_grad():
                    idx = 0
                    for module, A, B in lora_modules:
                        A.copy_(new_weights[idx])
                        B.copy_(new_weights[idx + 1])
                        idx += 2
    
                # Evaluate perturbed model
                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    perturbed_acc = metric(batch, self.peft_model, self.domain_map_fn)
                else:
                    perturbed_acc = metric(batch, self.peft_model)

                if perturbed_acc >= bound: # ! Successful if model accuracy after inflation is still within required bounds
                    success_count += 1
    
            # Restore nominal LoRA parameters
            with torch.no_grad():
                idx = 0
                for module, A, B in lora_modules:
                    A.copy_(self.LoRA_loc[idx])
                    B.copy_(self.LoRA_loc[idx + 1])
                    idx += 2
    
            # Compute Clopper-Pearson bound
            lower_bound, _ = proportion_confint( # ! gives a lower and upper bound confidence interval on the actually achieved bound certification
                success_count * self.smooth_cheat,
                iterations * self.smooth_cheat,
                alpha=delta,
                method='beta'
            )
            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate
    
            print(f"Inflate: {inflate:.1f}, Estimated success: {success_count/iterations:.4f}, Certified radius: {certified_radius:.4f}")
    
            if certified_radius > max_rad:
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break
    
        print(f"[LoRA] Max certified radius: {(0.01 * max_inflate) * max_rad:.4f} at inflate {max_inflate}")
        idx = 0
        for module, A, B in lora_modules:
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                module.A0 = module.lora_A.default.weight.detach().clone()
                module.B0 = module.lora_B.default.weight.detach().clone()
                #module.AS = max_inflate * self.LoRA_scale[idx]
                #module.BS = max_inflate * self.LoRA_scale[idx+1]
            idx += 2
        #attach_initial_lora_state(self.peft_model)
        self.LoRA_scale = [scale * max_inflate for scale in self.LoRA_scale]
        self.radius = max_rad * (0.01 * max_inflate) # multiplying by the constant in the initial density
        return max_rad

        
    def initialize_density_from_EWC(
        self,
        batch: DataLoader,
        metric: Callable,
        bound: float,
        steps: int,
        iterations: int,
        delta: float,
    ) -> float:
        """
        Initializes loc, scale (stddev), and radius for a Gaussian density
        using diagonal Fisher Information without flattening parameters.
        """
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize Fisher estimates (same shape as each param)
        fisher_diag = [torch.zeros_like(p) for p in params]
        
        for x, y in batch:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
    
            output = self.model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
    
            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher_diag[i] += p.grad.detach() ** 2
    
        # Average over batch size
        num_batches = len(batch)
        fisher_diag = [fd / num_batches for fd in fisher_diag]
    
        # Convert Fisher diagonals to stddevs (scale)
        epsilon = 1e-8
        self.loc = [p.detach().clone() for p in params]
        
        raw_scale = [1.0 / torch.sqrt(fd + epsilon) for fd in fisher_diag]

        # Find the global max across all tensors
        max_scale = max([s.max().item() for s in raw_scale])
        
        # Normalize so that max value becomes 0.01 (heuristic)
        self.scale = [s / (max_scale*100) for s in raw_scale]
        
        self.radius = 1e-8
        device = self.device
        model = self.model
        
        # Step 3: Assert nominal accuracy
        if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
            acc = metric(batch, model, self.domain_map_fn)
        else: 
            acc = metric(batch, model)
        assert acc > bound, f"Nominal accuracy {acc:.4f} does not exceed threshold {bound:.4f}"

        inflates = [1, 2, 5, 7.5, 10, 15, 25, 35, 50, 65] # hard coded, ask about better solution later
        max_rad = -1e6

        for inflate in inflates:
            
            # Step 4: Run smoothing simulations
            success_count = 0
            for _ in trange(iterations):
                new_params = []
                for loc, scale in zip(self.loc, self.scale):
                    noise = torch.normal(mean=0.0, std=scale * inflate)
                    new_params.append((loc + (noise)).reshape(loc.shape))
    
                with torch.no_grad():
                    for p, new_p in zip(model.parameters(), new_params):
                        p.copy_(new_p)

                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    perturbed_acc = metric(batch, model, self.domain_map_fn)
                else: 
                    perturbed_acc = metric(batch, model)
                if perturbed_acc >= bound:
                    success_count += 1
    
            # Step 5: Restore nominal parameters
            with torch.no_grad():
                for p, loc in zip(model.parameters(), self.loc):
                    p.copy_(loc)
    
            # Step 6: Compute Clopper-Pearson bound
            lower_bound, _ = proportion_confint(success_count * self.smooth_cheat, iterations * self.smooth_cheat, alpha=delta, method='beta')
            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate
    
            print(f"Certified L2 radius: {certified_radius:.4f}")

            if(certified_radius > max_rad):
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break

        print("Max rad: ", max_rad, " inflate: ", max_inflate)
        print("Last rad: ", certified_radius, " last inflate: ", inflate) 
        self.scale = [max_inflate * w for w in self.scale]
        self.radius = max_rad
        return certified_radius
        

    def initialize_density_from_buffer(self):
        return None

    def optimize_density_with_SGD(
        self,
        model: nn.Module,
        batch: DataLoader,
        metric: Callable,
        bound: float,
        steps: int,
        iterations: int,
        delta: float,
        lr: float = 1e-2,
        radius: float = 1.0,
        verbose: bool = True,
    ) -> float:
        """
        Learns a smoothing density using SGD to optimize the scale (stddev)
        under a fixed radius constraint (L2 = 1.0), then searches for the
        largest certifiable radius after training.
    
        Returns:
            float: certified radius using final optimized scale
        """
        self.model.eval()
        device = self.device
        params = [p for p in self.model.parameters() if p.requires_grad]
    
        # Step 1: Compute diagonal Fisher Information
        fisher_diag = [torch.zeros_like(p) for p in params]
        for x, y in batch:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            output = self.model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher_diag[i] += p.grad.detach() ** 2
        fisher_diag = [fd / len(batch) for fd in fisher_diag]
    
        # Step 2: Initialize loc and learnable log_scale
        epsilon = 1e-8
        self.loc = [p.detach().clone() for p in params]
        raw_scale = [1.0 / torch.sqrt(fd + epsilon) for fd in fisher_diag]
        max_scale = max([s.max().item() for s in raw_scale])
        init_scale = [s / (max_scale) for s in raw_scale]
    
        log_scales = [torch.nn.Parameter(torch.log(s + 1e-6)) for s in init_scale]
        optimizer = torch.optim.Adam(log_scales, lr=lr)
    
        for step in range(steps):
            all_losses = []
        
            for _ in range(iterations):
                noises = [torch.randn_like(log_s) * torch.exp(log_s) for log_s in log_scales]
                total_norm = torch.sqrt(sum([(n**2).sum() for n in noises]) + 1e-12)
                unit_noises = [10 * n * (radius / total_norm) for n in noises]
                perturbed_params = [loc + noise for loc, noise in zip(self.loc, unit_noises)]
        
                # Build a dictionary of perturbed param names and tensors
                named_params = dict(self.model.named_parameters())
                new_state_dict = {
                    name: perturbed for (name, _), perturbed in zip(named_params.items(), perturbed_params)
                }
        
                # Use functional_call instead of copying weights
                for x, y in batch:
                    x, y = x.to(device), y.to(device)
                    out = functional_call(self.model, new_state_dict, (x,))
                    log_probs = nn.functional.log_softmax(out, dim=1)
                    loss = -log_probs.gather(1, y.unsqueeze(1)).mean()
                    all_losses.append(loss)
        
            robustness_loss = torch.stack(all_losses).mean()
        
            optimizer.zero_grad()
            robustness_loss.backward()
            optimizer.step()
        
            print(f"[{step+1:02d}] Smoothed NLL Loss: {robustness_loss.item():.4f}")
        
          
        # Step 4: Save final scale
        self.scale = [torch.exp(log_s.detach()) for log_s in log_scales]
        # Find the global max across all tensors
        max_scale = max([s.max().item() for s in self.scale])
        
        # Normalize so that max value becomes 1.0
        self.scale = [s / (max_scale*100) for s in self.scale]
        
        # Step 5: Search best certifiable radius post-hoc
        inflates = [1, 2, 5, 7.5, 10, 15, 25, 35, 50, 65]
        max_rad = -1e6
    
        for inflate in inflates:
            success_count = 0
            for _ in range(iterations):
                noise = [torch.randn_like(s) * s * inflate for s in self.scale]
                new_params = [loc + n for loc, n in zip(self.loc, noise)]
    
                with torch.no_grad():
                    for p, new_p in zip(self.model.parameters(), new_params):
                        p.copy_(new_p)

                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    perturbed_acc = metric(batch, model, self.domain_map_fn)
                else: 
                    perturbed_acc = metric(batch, model)
                if perturbed_acc >= bound:
                    success_count += 1
    
            with torch.no_grad():
                for p, loc in zip(self.model.parameters(), self.loc):
                    p.copy_(loc)
    
            lower_bound, _ = proportion_confint(success_count * self.smooth_cheat, iterations * self.smooth_cheat, alpha=delta, method="beta")
            certified_radius = norm.ppf(lower_bound) * inflate
    
            print(f"Inflate {inflate:.1f}: Certified Radius = {certified_radius:.4f}")
    
            if certified_radius > max_rad:
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break
    
        print("Max rad: ", max_rad, " inflate: ", max_inflate)
        print("Last rad: ", certified_radius, " last inflate: ", inflate) 
        # self.scale = [max_inflate * w for w in widths]
        self.scale = [max_inflate * w for w in self.scale] # ? I think it should be this rather than the original above
        self.radius = max_rad
        return max_rad

        
    def initialize_density_from_interval(
        self,
        bounded_model: IntervalBoundedModel,
        batch: DataLoader,
        metric: Callable,
        bound: float,
        steps: int,
        iterations: int,
        delta: float,
    ) -> float:
        """
        Initializes self.loc and self.scale from interval bounds and certifies the smoothing radius.
        """
        
        inflates = [1, 2, 5, 7.5, 10, 15, 25, 35, 50, 65] # hard coded, ask about better solution later
        max_rad = -1e6
        
        self.loc = [p.detach().clone().to(torch.float32).to(self.device) for p in self.model.parameters()]
        widths = [
                (param_u - param_l).to(torch.float32).to(self.device)
                for param_l, param_u in zip(bounded_model.param_l, bounded_model.param_u)
            ]
        
        for inflate in inflates:
            device = self.device
            model = self.model

            test_scale = [w * inflate for w in widths]
            
            # Step 3: Assert nominal accuracy
            if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                acc = metric(batch, model, self.domain_map_fn)
            else:
                acc = metric(batch, model)
            assert acc >= bound, f"Nominal accuracy {acc:.4f} does not exceed threshold {bound:.4f}"
    
            # Step 4: Run smoothing simulations
            success_count = 0
            for _ in trange(iterations):
                new_params = []
                for loc, scale in zip(self.loc, test_scale):
                    noise = torch.normal(mean=0.0, std=scale)
                    new_params.append((loc + noise).reshape(loc.shape))
    
                with torch.no_grad():
                    for p, new_p in zip(model.parameters(), new_params):
                        p.copy_(new_p)
    
                if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
                    perturbed_acc = metric(batch, model, self.domain_map_fn)
                    #print("[LOG]: Acc of perturbed model (DIL) = ", perturbed_acc)
                else: 
                    perturbed_acc = metric(batch, model)
                    #print("[LOG]: Acc of perturbed model = ", perturbed_acc)
                if perturbed_acc >= bound:
                    success_count += 1
    
            # Step 5: Restore nominal parameters
            with torch.no_grad():
                for p, loc in zip(model.parameters(), self.loc):
                    p.copy_(loc)
    
            # Step 6: Compute Clopper-Pearson bound
            lower_bound, _ = proportion_confint(success_count * self.smooth_cheat, iterations * self.smooth_cheat, alpha=delta, method='beta')
            print("Estimated success: ", success_count/iterations, " Lower bound: ", lower_bound)
            lower_bound = float(lower_bound)
            certified_radius = norm.ppf(lower_bound) * inflate
    
            print(f"Certified L2 radius: {certified_radius:.4f}")

            if(certified_radius > max_rad):
                max_rad = certified_radius
                max_inflate = inflate
            else:
                break

        print("Max rad: ", max_rad, " inflate: ", max_inflate)
        print("Last rad: ", certified_radius, " last inflate: ", inflate) 
        self.scale = [max_inflate * w for w in widths]
        self.radius = max_rad
        return certified_radius

        
    def compute_rashomon_set(
        self,
        dataset: torch.utils.data.Dataset,
        callback: Callable | None = None,
        use_outer_bbox: bool = True,
        context_id: int | None = None,
    ) -> None:
        """
        Compute the Rashomon set on the given data.
        """
        self.compute_rashomon_set_helper(dataset, callback, use_outer_bbox, context_id)
        
        
    def compute_rashomon_set_helper(
        self,
        dataset: torch.utils.data.Dataset,
        callback: Callable | None = None,
        use_outer_bbox: bool = True,
        context_id: int | None = None,
    ) -> None:
        """
        Compute the Rashomon set on the given data - helper
        """
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=self.n_certificate_samples, shuffle=True
        )
        X, y = next(iter(dl))
        X, y = X.to(self.device), y.to(self.device)

        if hasattr(self, "domain_map_fn") and self.domain_map_fn is not None:
            y = self.domain_map_fn(y)

        task_acc = (self.model(X).argmax(dim=1) == y).float().mean().item()
        
        if isinstance(self.model[-1], utils.InContextHead):
            self.model[-1].set_context(context_id)
            model = self.model[:-1]
            context_mask = self.model[-1].mask
            if(self.peft_model.model[-1] is not None):
                self.peft_model.model[-1].set_context(context_id)
        else:
            model = self.model
            context_mask = None

        if self.min_acc_increment and self.min_acc_limit:
            min_acc_limit = min(task_acc - self.min_acc_increment, self.min_acc_limit)
        elif self.min_acc_increment:
            min_acc_limit = task_acc - self.min_acc_increment
        elif self.min_acc_limit:
            min_acc_limit = self.min_acc_limit
        else:
            raise ValueError(
                "min_acc_limit or min_acc_increment must be set to compute the Rashomon set."
            )
        dl = torch.utils.data.DataLoader(dataset, batch_size=10_000)
        if(self.density_strategy == "fim_init"):
            self.initialize_density_from_EWC(dl, self.smooth_metric, 
                                                  self.smooth_bound, self.smooth_steps, self.smooth_iterations, 
                                                  self.smooth_delta)
        elif(self.density_strategy == "sgd_opt"):
            self.optimize_density_with_SGD(dl, self.smooth_metric, 
                                                  self.smooth_bound, self.smooth_steps, self.smooth_iterations, 
                                                  self.smooth_delta)
            
        elif(self.density_strategy == "interval_init"):
            self.initialize_density_from_interval(self.pre_computed_bound, dl, self.smooth_metric, 
                                                  self.smooth_bound, self.smooth_steps, self.smooth_iterations, 
                                                  self.smooth_delta)
            
        elif(self.density_strategy == "LoRA"):
            self.initialize_density_from_LoRA(dl, self.smooth_metric, 
                                                  self.smooth_bound, self.smooth_steps, self.smooth_iterations, 
                                                  self.smooth_delta)
    def _train_step(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        regulariser: BaseRegulariser = None,
        **kwargs: dict,
    ) -> tuple[float, float]:
    
        if(self.peft_model is not None):
            assert(type(model) == peft.tuners.lora.model.LoraModel)
            self.peft_model.train()
            
        model.train()
        inputs, targets = X.to(self.device), y.to(self.device)

        # As in variational inference, we should sample a parameter for each update...
        #new_params = []
        #for loc, scale in zip(self.loc, test_scale):
        #    noise = torch.normal(mean=0.0, std=scale)
        #    new_params.append((loc + noise).reshape(loc.shape))
                    
        if hasattr(self, "domain_map_fn") and self.domain_map_fn:
            targets = self.domain_map_fn(targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if regulariser is not None:
            loss += regulariser(model=model, inputs=inputs)
        loss.backward()
        optimizer.step()

        self._project_parameters(model, inputs, targets)

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
        assert self.radius, "Bounds must have been computed prior to calling train."

        self._project_parameters(model)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        val_acc = None

        if (
            self.paradigm == "TIL"
            and (context := kwargs.get("context_id", None)) is not None
            and isinstance(model[-1], utils.InContextHead)
        ):
            model[-1].set_context(context)
            if(self.peft_model is not None):
                #print("PEFT USED")
                self.peft_model.model[-1].set_context(kwargs["context_id"])

        for epoch in (pbar := tqdm(range(epochs), desc="Training Epochs")):
            model.train()
            if(self.peft_model is not None):
                self.peft_model.train()
            for inputs, targets in train_loader:
                if(self.peft_model is None):
                    loss, _ = self._train_step(
                        model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                    )
                else:
                    loss, _ = self._train_step(
                        self.peft_model, inputs, targets, optimizer, loss_fn, regulariser, **kwargs
                    )
            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "val_acc": f"{val_acc:.4f}" if val_acc is not None else None,
                    "proj": self._last_projection,
                }
            )
            if(self.peft_model is not None):
                
                val_loss, val_acc = self._validate_model(
                    model,
                    val_loader,
                    loss_fn,
                    regulariser,
                    domain_map_fn=self.domain_map_fn
                    if hasattr(self, "domain_map_fn")
                    else None,
                )
            else:
                val_loss, val_acc = self._validate_model(
                    self.peft_model,
                    val_loader,
                    loss_fn,
                    regulariser,
                    domain_map_fn=self.domain_map_fn
                    if hasattr(self, "domain_map_fn")
                    else None,
                )                

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "val_acc": f"{val_acc:.4f}" if val_acc is not None else None,
                    "proj": self._last_projection,
                }
            )

            if early_stopping:
                stopping, epochs_no_improve = self.check_early_stopping(
                    curr_loss=val_loss,
                    prev_loss=best_val_loss if best_val_loss != float("inf") else None,
                    patience=kwargs.get("patience", 5),
                    no_improvement=epochs_no_improve,
                )

                if stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
                else:
                    best_model_state = model.state_dict()
        return model

    @torch.no_grad()
    def _test(
        self, test_loader: DataLoader, loss_fn: Callable, **kwargs: dict
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        last_task = kwargs.get("last_task", False)

        if (
            self.paradigm == "TIL"
            and kwargs.get("context_id", None) is not None
            and isinstance(self.model[-1], utils.InContextHead)
        ):
            self.model[-1].set_context(kwargs["context_id"])
            if(self.peft_model is not None):
                #print("PEFT USED")
                self.peft_model.model[-1].set_context(kwargs["context_id"])

        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if hasattr(self, "domain_map_fn"):
                targets = self.domain_map_fn(targets)

            if(self.peft_model is None):
                #print("NO PEFT")
                preds = []
                for _ in range(self.smooth_infer_samps):
                    model_copy = copy.deepcopy(self.model).to(self.device)
                    model_copy.eval()
                    
                    with torch.no_grad():
                        for param, loc, scale in zip(model_copy.parameters(), model_copy.parameters(), self.scale):
                            noise = torch.normal(mean=0.0, std=scale)
                            param.copy_(loc + noise)
        
                        out = model_copy(inputs.to(self.device))
                        pred = out.argmax(dim=1)
                        one_hot = F.one_hot(pred, num_classes=self.num_classes).float()
                        preds.append(one_hot)
            
            else:
                #print("PEFT USED")
                preds = []
                for _ in range(self.smooth_infer_samps):
                    model_copy = copy.deepcopy(self.model).to(self.device)
                    model_copy.eval()
                
                    with torch.no_grad():
                        i = 0  # index into self.LoRA_scale
                        for module in model_copy.modules():
                            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                                scale_A = self.LoRA_scale[i]
                                scale_B = self.LoRA_scale[i + 1]
                                loc_A = self.LoRA_loc[i]
                                loc_B = self.LoRA_loc[i + 1]
                
                                noise_A = torch.normal(mean=0.0, std=scale_A).to(loc_A.device)
                                noise_B = torch.normal(mean=0.0, std=scale_B).to(loc_B.device)
                
                                module.lora_A.default.weight.copy_(loc_A + noise_A)
                                module.lora_B.default.weight.copy_(loc_B + noise_B)
                
                                i += 2  # step to next pair
                
                        out = model_copy(inputs.to(self.device))
                        pred = out.argmax(dim=1)
                        one_hot = F.one_hot(pred, num_classes=self.num_classes).float()
                        preds.append(one_hot)
                        
            outputs = torch.stack(preds, dim=0).mean(dim=0)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        return round(avg_loss, 4), round(accuracy, 4)
