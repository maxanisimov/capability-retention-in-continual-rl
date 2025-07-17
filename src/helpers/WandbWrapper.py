import wandb
import copy
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np

from src.trainer import (
    BaseTrainer,
    SimpleTrainer,
    BufferTrainer,
    FisherTrainer,
    InterContiNetTrainer,
    IntervalTrainer,
    SIBufferTrainer,
    SITrainer,
)
from src.regulariser import L2Regulariser, MultiRegulariser, UnbiasRegulariser


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
        **static_kwargs: dict,
    ):
        self.trainer_class = trainer_class
        self.model = model
        self.train_tasks = train_tasks
        self.val_tasks = val_tasks
        self.test_tasks = test_tasks
        self.static_kwargs = static_kwargs

    def run(self):
        with wandb.init(settings=wandb.Settings(_service_wait=60)) as run:
            config = unflatten_dict(wandb.config)
            print(config)

            # Combine static args with wandb's dynamic config
            # wandb config takes precedence
            full_config = {**self.static_kwargs, **config}

            simple_train_args = full_config["simple_train"]
            init_args = full_config["init"]
            train_args = full_config["train"]

            # --- Instantiate and Train ---
            # Instantiate the trainer with its specific init args
            l2 = L2Regulariser(lmbd=train_args["l2_lambda"])
            unbias = UnbiasRegulariser(lmbd=train_args["unbias_lambda"])
            regulariser = MultiRegulariser([unbias, l2])

            st = SimpleTrainer(self.model, seed=init_args["seed"])
            st.train(
                self.train_tasks[0],
                self.val_tasks[0],
                epochs=simple_train_args["epochs"],
                batch_size=simple_train_args["batch_size"],
                regulariser=regulariser,
                lr=simple_train_args["lr"],
                weight_decay=simple_train_args["weight_decay"],
            )
            result = st.test(self.test_tasks[0:1])
            if result[-1][1] < 0.8:
                print("Initial task accuracy too low.")
                log_data = {
                    "final_num_tasks": 0,
                    "final_avg_accuracy": 0,
                    "second_task_accuracy": 0,
                    "final_avg_loss": 9999,
                    "final_total_accuracy": 0,
                }
                wandb.log(log_data)
                return

            save_model = copy.deepcopy(st.model)

            trainer = self.trainer_class(model=save_model, **init_args)

            if type(trainer) is BufferTrainer:
                raise NotImplementedError()
            if type(trainer) is FisherTrainer:
                raise NotImplementedError()
            if type(trainer) is InterContiNetTrainer:
                for i, (train, val) in enumerate(
                    zip(self.train_tasks[1:], self.val_tasks[1:]), start=1
                ):
                    trainer.compute_rashomon_set(
                        self.test_tasks[i - 1],
                    )

                    trainer.train(
                        train,
                        val,
                        epochs=20,
                        batch_size=256,
                        early_stopping=True,
                        patience=10,
                        lr=train_args["lr"],
                        weight_decay=train_args["weight_decay"],
                        regulariser=regulariser,
                    )
                    results = trainer.test(self.test_tasks[0 : i + 1])
                    target_acc = min(max(results[-1][1] - trainer.min_acc_increment, results[-1][1] / 2), trainer.min_acc_limit)
                    trainer.min_acc_limit = target_acc
                    if not all(res[1] for res in results):
                        print("Catastrophic Forgetting occurred.")
                        break

                accuracies = [res[1] for res in results]
                avg_accuracy = np.mean(accuracies)

                losses = [res[0] for res in results]
                avg_loss = np.mean(losses)

                log_data = {
                    "final_num_tasks": len(results),
                    "final_avg_accuracy": avg_accuracy,
                    "second_task_accuracy": accuracies[1] if len(accuracies) > 1 else 0,
                    "final_avg_loss": avg_loss,
                    "final_total_accuracy": np.sum(accuracies),
                }
            if type(trainer) is IntervalTrainer:
                for i, (train, val) in enumerate(
                    zip(self.train_tasks[1:], self.val_tasks[1:]), start=1
                ):
                    trainer.compute_rashomon_set(
                        self.test_tasks[i - 1],
                    )

                    trainer.train(
                        train,
                        val,
                        epochs=train_args["epochs"],
                        batch_size=train_args["batch_size"],
                        lr=train_args["lr"],
                        weight_decay=train_args["weight_decay"],
                        regulariser=regulariser,
                    )
                    results = trainer.test(self.test_tasks[0 : i + 1])
                    if not all(res[1] for res in results):
                        print("Catastrophic Forgetting occurred.")
                        break

                accuracies = [res[1] for res in results]
                avg_accuracy = np.mean(accuracies)

                losses = [res[0] for res in results]
                avg_loss = np.mean(losses)

                log_data = {
                    "final_num_tasks": len(results),
                    "final_avg_accuracy": avg_accuracy,
                    "second_task_accuracy": accuracies[1] if len(accuracies) > 1 else 0,
                    "final_avg_loss": avg_loss,
                    "final_total_accuracy": np.sum(accuracies),
                }
            if type(trainer) is SIBufferTrainer:
                raise NotImplementedError()
            if type(trainer) is SITrainer:
                for i, (train, val) in enumerate(
                    zip(self.train_tasks[1:], self.val_tasks[1:]), start=1
                ):
                    loader = DataLoader(
                        train,
                        batch_size=train_args["si_batch_size"],
                        shuffle=True,
                        generator=torch.Generator().manual_seed(init_args["seed"]),
                    )

                    samples = next(iter(loader))
                    trainer.compute_rashomon_set(
                        self.test_tasks[i - 1],
                        prune_prop=train_args["prune_prop"],
                        si_batch=samples,
                        si_steps=train_args["si_steps"],
                    )
                    assert trainer.test(self.test_tasks[0 : i + 1])[-1][1] == 0, (
                        "Prior last task performance needs to be zero."
                    )

                    trainer.train(
                        train,
                        val,
                        epochs=20,
                        batch_size=256,
                        early_stopping=True,
                        patience=10,
                        lr=train_args["lr"],
                        weight_decay=train_args["weight_decay"],
                        regulariser=regulariser,
                    )
                    results = trainer.test(self.test_tasks[0 : i + 1])
                    if not all(res[1] for res in results):
                        print("Catastrophic Forgetting occurred.")
                        break

                accuracies = [res[1] for res in results]
                avg_accuracy = np.mean(accuracies)

                losses = [res[0] for res in results]
                avg_loss = np.mean(losses)

                log_data = {
                    "final_num_tasks": len(results),
                    "final_avg_accuracy": avg_accuracy,
                    "second_task_accuracy": accuracies[1] if len(accuracies) > 1 else 0,
                    "final_avg_loss": avg_loss,
                    "final_total_accuracy": np.sum(accuracies),
                }

            # Log the final validation metrics from the last epoch
            wandb.log(log_data)
