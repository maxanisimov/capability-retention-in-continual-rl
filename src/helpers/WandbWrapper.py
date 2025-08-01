import wandb
import copy
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
from typing import Callable

from src.trainer import (
    BaseTrainer,
    SimpleTrainer,
    BufferTrainer,
    FisherTrainer,
    InterContiNetTrainer,
    IntervalTrainer,
    SIBufferTrainer,
    SITrainer,
    EWCTrainer,
    LwFTrainer,
)
from src.regulariser import (
    L2Regulariser,
    MultiRegulariser,
    UnbiasRegulariser,
)
from src.data_utils import get_context_sets
from src.utils.general import set_seed, InContextHead
from src.models import get_fully_connected_model


def unflatten_dict(d: dict) -> dict:
    """Converts a flat, dot-separated dictionary into a nested one."""
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


class WandbTrainerWrapper:
    def __init__(
        self,
        trainer_class: BaseTrainer,
        model: nn.Module,
        train_tasks: list[Dataset],
        val_tasks: list[Dataset],
        test_tasks: list[Dataset],
        domain_map_fn: Callable = None,
        seed: int = 42,
        **static_kwargs: dict,
    ):
        self.trainer_class = trainer_class
        self.model = model
        self.train_tasks = train_tasks
        self.val_tasks = val_tasks
        self.test_tasks = test_tasks
        self.domain_map_fn = domain_map_fn
        self.static_kwargs = static_kwargs
        self.seed = seed
        set_seed(self.seed)

    def run(
        self,
        project="certified-continual-learning",
        tags: list[str] = [],
        config: dict = None,
        unbias_domain: list[torch.Tensor] = None,
    ):
        with wandb.init(project=project, config=config, tags=tags) as run:
            wandb.log({"seed": self.seed})
            config = unflatten_dict(wandb.config)

            # Combine static args with wandb's dynamic config
            # wandb config takes precedence
            full_config = {**self.static_kwargs, **config}

            init_args = full_config["init"]
            train_args = full_config["train"]
            columns = [f"Test Task {i}" for i in range(len(self.test_tasks))]
            rows = [f"Task {i}" for i in range(len(self.test_tasks))] + ["Certificates"]

            if config["ours"]:
                # --- Instantiate and Train ---
                # Instantiate the trainer with its specific init args
                l2 = L2Regulariser(lmbd=train_args["l2_lambda"])
                unbias = UnbiasRegulariser(
                    lmbd=train_args["unbias_lambda"], unbias_domain=unbias_domain
                )
                regulariser = MultiRegulariser([unbias, l2])
                trainer = self.trainer_class(
                    model=self.model,
                    domain_map_fn=self.domain_map_fn
                    if init_args["paradigm"] == "DIL"
                    else None,
                    **init_args,
                )

                accuracy_matrix = []
                if type(trainer) is BufferTrainer:
                    raise NotImplementedError()
                if type(trainer) is FisherTrainer:
                    raise NotImplementedError()
                if type(trainer) is InterContiNetTrainer:
                    raise NotImplementedError()
                if type(trainer) is IntervalTrainer:
                    for i, (train, val, test) in enumerate(
                        zip(self.train_tasks, self.val_tasks, self.test_tasks)
                    ):
                        trainer.train(
                            train,
                            val,
                            epochs=train_args["epochs"],
                            batch_size=train_args["batch_size"],
                            lr=train_args["lr"],
                            weight_decay=train_args["weight_decay"],
                            regulariser=regulariser,
                            context_id=i if trainer.paradigm == "TIL" else None,
                        )
                        results = trainer.test(
                            self.test_tasks,
                            context_list=list(range(len(self.test_tasks)))
                            if trainer.paradigm == "TIL"
                            else [None] * len(self.test_tasks),
                        )
                        accs = [res[1] for res in results]
                        if i == 0 and accs[0] < 0.7:
                            print("Initial Accuracy too low.")
                            wandb.finish(1)
                            return
                        if i < len(self.test_tasks) - 1:
                            trainer.compute_rashomon_set(
                                test,
                                context_id=i if trainer.paradigm == "TIL" else None,
                            )
                        accuracy_matrix.append(accs)

                if type(trainer) is SIBufferTrainer:
                    raise NotImplementedError()
                if type(trainer) is SITrainer:
                    raise NotImplementedError()

                avg_accuracy = np.mean(accs)

                losses = [res[0] for res in results]
                avg_loss = np.mean(losses)

                log_data = {
                    "final_num_tasks": len(results),
                    "final_avg_accuracy": avg_accuracy,
                    "second_task_accuracy": accs[1] if len(accs) > 1 else 0,
                    "final_avg_loss": avg_loss,
                    "final_total_accuracy": np.sum(accs),
                }

                # Log the final validation metrics from the last epoch
                accuracy_matrix.append(trainer.final_certificates + [0])
                wandb.log(
                    {
                        "accuracy_matrix": wandb.Table(
                            data=accuracy_matrix, columns=columns, rows=rows
                        )
                    }
                )
                wandb.log(log_data)

            # Run benchmarks
            bt_trainers = {
                "ewc": EWCTrainer,
                "sgd": SimpleTrainer,
                "lwf": LwFTrainer,
                "icn": InterContiNetTrainer,
            }
            bconfig = config.get("benchmark_config", None)
            for benchmark in config.get("benchmarks", {}).keys():
                b_acc_matrix = []
                print(f"Running benchmark: {benchmark}.")
                bconfig = config["benchmarks"][benchmark] or {}
                bt = bt_trainers[benchmark](
                    model=self.model,
                    seed=self.seed,
                    paradigm=init_args["paradigm"],
                    domain_map_fn=self.domain_map_fn
                    if init_args["paradigm"] == "DIL"
                    else None,
                    context_sets=get_context_sets(self.test_tasks),
                    **bconfig,
                )
                for i, (train, val, test) in enumerate(
                    zip(self.train_tasks, self.val_tasks, self.test_tasks),
                ):
                    bt.train(
                        train,
                        val,
                        context_id=i
                        if init_args["paradigm"] == "TIL" or type(bt) is LwFTrainer
                        else None,
                        fisher_batch=bconfig.get("fisher_batch", 64),
                        lr=bconfig.get("lr", 0.01),
                        weight_decay=bconfig.get("weight_decay", 0),
                        epochs=bconfig.get("epochs", 5),
                        batch_size=bconfig.get("batch_size", 128)
                    )
                    results = bt.test(
                        self.test_tasks,
                        context_list=list(range(len(self.test_tasks)))
                        if init_args["paradigm"] == "TIL"
                        else [None] * len(self.test_tasks),
                    )
                    if benchmark == "icn" and not i and results[0][1] < 0.65:
                        wandb.finish(1)
                        return
                    if benchmark == "icn" and i < len(self.train_tasks) - 1:
                        target_acc = max(
                            results[i][1] - bt.min_acc_increment, results[i][1] / 2
                        )
                        bt.min_acc_limit = target_acc
                        print(target_acc)
                        bt.compute_rashomon_set(
                            test,
                            context_id=i if init_args["paradigm"] == "TIL" else None,
                            lr=bconfig["lid_lr"],
                            batch_size=init_args["n_certificate_samples"],
                            epochs=1000,
                        )
                    b_acc_matrix.append([res[1] for res in results])

                if benchmark == "icn":
                    b_acc_matrix.append(bt.final_certificates + [0])

                wandb.log(
                    {
                        f"accuracy_matrix_{benchmark}": wandb.Table(
                            data=b_acc_matrix,
                            columns=columns,
                            rows=rows[:-1] if benchmark != "icn" else rows,
                        )
                    }
                )

            wandb.finish()
