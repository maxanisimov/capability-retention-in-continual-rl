import wandb
import torch

from src.trainer import AGEMTrainer
from src.data_utils import get_mnist_tasks, _extract_targets, get_context_sets
from src.utils.general import InContextHead
from src import models

from configs import MNIST_AGEM_CONFIG as CONFIG

# HYPERPARAMS
""" TIL
MNIST = {
    "batch_size": 256,
    "epochs": 3,
    "lr": 0.001,
    "weight_decay": 0,
    "unbias_lambda": 0.01,
    "l2_lambda": 0.01
}
"""

if __name__ == "__main__":
    for paradigm in ["TIL", "DIL", "CIL"]:
        for i in range(5, 15):
            failed = False
            with wandb.init(
                project="certified-continual-learning",
                reinit=True,
                tags=["final_mnist_buffer", "buffer_agem", f"buffer_{paradigm.lower()}"],
            ):
                def domain_map_fn(labels: torch.Tensor) -> torch.Tensor:
                    """Map the global label to the in context label."""
                    return labels % 2
                wandb.log({"seed": i})
                SEED = i
                train_tasks, _, test_tasks = get_mnist_tasks(seed=SEED, train_val_split_ratio=0.3, emnist=True)
                context_sets = get_context_sets(test_tasks)
                head = InContextHead(context_sets, 10, device="cuda")
                head.set_context(0)
                model = models.get_mnist_model(device="cuda", output_dim=2 if paradigm == "DIL" else 10, seed=SEED, head = head if paradigm=="TIL" else None)
                print(
                    f"Tasks: {[torch._unique(_extract_targets(train))[0].tolist() for train in train_tasks]}"
                )

                agem_trainer = AGEMTrainer(
                    model,
                    memory_samples=3750,
                    paradigm=paradigm,
                    seed=SEED,
                    domain_map_fn=domain_map_fn if paradigm == "DIL" else None
                )

                acc_matrix = []
                for i, (train, test) in enumerate(
                    zip(train_tasks, test_tasks)
                ):
                    agem_trainer.train(
                        train,
                        test,
                        batch_size=CONFIG["batch_size"],
                        epochs=CONFIG["epochs"],
                        lr=CONFIG["lr"],
                        weight_decay=CONFIG["weight_decay"],
                        context_id=i if paradigm == "TIL" else None,
                        val_freq=int(len(train) / CONFIG["batch_size"]) - 1
                    )
                    results = agem_trainer.test(
                        test_tasks,
                        context_list=list(range(len(test_tasks)))
                        if paradigm == "TIL"
                        else [None] * len(test_tasks),
                    )
                    accs = [res[1] for res in results]
                    if not i and accs[0] < 0.7:
                        wandb.finish(1)
                        failed = True
                        break
                    acc_matrix.append(accs)

                if not failed:
                    columns = [f"Test Task {i}" for i in range(len(test_tasks))]
                    rows = [f"Task {i}" for i in range(len(test_tasks))]
                    wandb.log(
                        {
                            "accuracy_matrix": wandb.Table(
                                data=acc_matrix, columns=columns, rows=rows
                            )
                        }
                    )
                    wandb.finish()