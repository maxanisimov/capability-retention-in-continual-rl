import wandb
import argparse

from src.wandb_configs import SWEEP_CONFIGS
from src.trainer import SITrainer, IntervalTrainer
from src.helpers.WandbWrapper import WandbTrainerWrapper
from src.data_utils import get_mnist_tasks
from src.models import get_mnist_model

# (Paste SWEEP_CONFIGS and TRAINER_MAPPING dictionaries here)

TRAINER_MAPPING = {
    "SITrainer": SITrainer,
    "IntervalTrainer": IntervalTrainer
}

def main():
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run a W&B sweep for a specific trainer.")
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=["SITrainer", "IntervalTrainer"],
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

    print(f"🚀 Starting sweep for trainer: {trainer_name}")
    sweep_id = wandb.sweep(sweep=sweep_config, project="my-multi-trainer-project")
    
    if args.dataset == "MNIST":
        train_tasks, val_tasks, test_tasks = get_mnist_tasks()
        model = get_mnist_model()

    wrapper = WandbTrainerWrapper(
        trainer_class=trainer_class,
        model=model,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
    )

    # --- 4. Run the agent ---
    def train_fn():
        return wrapper.run()
    wandb.agent(sweep_id, function=train_fn, count=500)


if __name__ == "__main__":
    main()