import wandb
import argparse
import torch
import random

from src.wandb_configs import SWEEP_CONFIGS
from src.trainer import SITrainer, IntervalTrainer, InterContiNetTrainer
from src.helpers.WandbWrapper import WandbTrainerWrapper
from src.data_utils import get_mnist_tasks, get_context_sets
from src.models import get_mnist_model
from src.utils.general import InContextHead

# (Paste SWEEP_CONFIGS and TRAINER_MAPPING dictionaries here)

TRAINER_MAPPING = {
    "SITrainer": SITrainer,
    "IntervalTrainer": IntervalTrainer,
    "InterContiNetTrainer": InterContiNetTrainer
}

def main():
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run a W&B sweep for a specific trainer.")
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=["SITrainer", "IntervalTrainer", "InterContiNetTrainer"],
        help="The name of the trainer configuration to use for the sweep."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MNIST"],
        help="The name of the dataset to be used."
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        required=True,
        choices=["TIL", "CIL", "DIL"],
        help="The continual learning paradigm."
    )
    args = parser.parse_args()
    trainer_name = args.trainer

    # --- 2. Select the correct config and class ---
    sweep_config = SWEEP_CONFIGS[args.dataset][trainer_name][args.paradigm]
    trainer_class = TRAINER_MAPPING[trainer_name]
    SEED = random.randint(0, 1000)

    print(f"🚀 Starting sweep for trainer: {trainer_name}")
    sweep_id = wandb.sweep(sweep=sweep_config, project="my-multi-trainer-project")
    
    domain_map_fn = None
    if args.dataset == "MNIST":
        train_tasks, val_tasks, test_tasks = get_mnist_tasks(seed=SEED)
        context_sets = get_context_sets(test_tasks)
        output_dim = 2 if args.paradigm == " DIL" else 10
        head = InContextHead(context_sets, 10, device="cuda") if args.paradigm == "TIL" else None
        if head:
            head.set_context(0)
        model = get_mnist_model(head=head, device="cuda", seed=SEED, output_dim=output_dim)

        if args.paradigm == "DIL":
            def mapping_fn(labels: torch.Tensor) -> torch.Tensor:
                """Map the global label to the in context label."""
                return labels % 2
            domain_map_fn = mapping_fn 

    if args.dataset == "CIFAR":
        raise NotImplementedError()

    wrapper = WandbTrainerWrapper(
        trainer_class=trainer_class,
        model=model,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
        domain_map_fn=domain_map_fn,
        seed=SEED
    )

    # --- 4. Run the agent ---
    def train_fn():
        return wrapper.run(tags=[args.trainer])
    wandb.agent(sweep_id, function=train_fn, count=10)


if __name__ == "__main__":
    main()