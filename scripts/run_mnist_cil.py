from src.helpers.WandbWrapper import WandbTrainerWrapper
from src.trainer import IntervalTrainer
from src.models import get_mnist_model
from src.data_utils import get_mnist_tasks
from src.utils.general import set_seed
from configs import MNIST_IT_CONFIG as CONFIG
import wandb
import torch

config = {
    "ours": True,
    "init.projection_strategy": CONFIG["projection_strategy"],
    "init.n_certificate_samples": CONFIG["n_certificate_samples"],
    "init.min_acc_limit": CONFIG["min_acc_limit"],
    "init.min_acc_increment": CONFIG["min_acc_increment"],
    "init.paradigm": "CIL",
    "init.n_iters": CONFIG["n_iters"],
    "init.primal_learning_rate": CONFIG["primal_learning_rate"],
    "init.dual_learning_rate": CONFIG["dual_learning_rate"],
    "init.penalty_coefficient": CONFIG["penalty_coefficient"],
    "init.checkpoint": CONFIG["checkpoint"],
    "train.l2_lambda": CONFIG["l2_lambda"],
    "train.unbias_lambda": CONFIG["unbias_lambda"],
    "train.lr": CONFIG["lr"],
    "train.weight_decay": CONFIG["weight_decay"],
    "train.epochs": CONFIG["epochs"],
    "train.batch_size": CONFIG["batch_size"],
    "benchmarks": {
        "ewc": {"lmbd": 1e6, "fisher_batch": 64},
        "sgd": None,
        "lwf": {"lmbd": 0.05, "temp": 2},
        "icn": {"lr": 0.01, "batch_size": 128, "epochs": 30, "lid_lr": 100},
    },
}


def main():
    tag = "final_mnist_cil_new"
    total_seeds = 100
    benchmark_tags = [
        f"final_mnist_cil_{bench}" for bench in config["benchmarks"].keys()
    ]
    # Check if there are existing runs with the same tag
    api = wandb.Api()
    existing_runs = list(
        api.runs(
            "certified-continual-learning",
            {"tags": {"$in": [tag]}, "state": "finished"},
        )
    )
    n_runs = len(existing_runs)
    print(f"Found {n_runs} existing runs with tag '{tag}'")
    print(f"Initializing the remaining {total_seeds - n_runs} runs...")

    for i in range(n_runs, total_seeds):
        set_seed(i)
        config["init.seed"] = i
        train_tasks, val_tasks, test_tasks = get_mnist_tasks(seed=config["init.seed"])
        model = get_mnist_model(seed=config["init.seed"], device="cuda")
        wrapper = WandbTrainerWrapper(
            trainer_class=IntervalTrainer,
            model=model,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            test_tasks=test_tasks,
            seed=config["init.seed"],
        )
        wrapper.run(
            project="certified-continual-learning",
            tags=["final_mnist_new"]
            + ([tag] if config["ours"] else [])
            + benchmark_tags,
            config=config,
            unbias_domain=[
                torch.zeros(1, 1, 28, 28, device="cuda"),
                torch.ones(1, 1, 28, 28, device="cuda"),
            ],
        )


if __name__ == "__main__":
    main()
