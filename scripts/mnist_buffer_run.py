import wandb
import torch

from src.trainer.BufferTrainer import BufferTrainer
from src.data_utils import (
    get_mnist_tasks,
    _extract_targets,
    get_context_sets,
    create_holdout_set,
)
from src.utils.general import InContextHead, print_colored
from src import models
from src.buffer import MultiTaskBuffer

from src.regulariser import UnbiasRegulariser, L2Regulariser, MultiRegulariser

from configs import MNIST_BUFFER_CONFIG as CONFIG

def run_buffer(buffer_size: int, seed: int, config: wandb.config, paradigm="TIL"):
    device = "cuda"
    SMALL = 1000
    MEDIUM = 5000
    LARGE = 15000
    def domain_map_fn(labels: torch.Tensor) -> torch.Tensor:
        """Map the global label to the in context label."""
        return labels % 2
    SEED = seed
    CONFIG = config
    train_tasks, _, test_tasks = get_mnist_tasks(seed=SEED, emnist=True, train_val_split_ratio=1)
    
    context_sets = get_context_sets(test_tasks)
    head = InContextHead(context_sets, 10, device=device)
    head.set_context(0)
    model = models.get_mnist_model(device=device, output_dim=10, seed=SEED, head = head if paradigm=="TIL" else None)
    print(
        f"Tasks: {[torch._unique(_extract_targets(train))[0].tolist() for train in train_tasks]}"
    )

    unbias = UnbiasRegulariser(
    lmbd=CONFIG["unbias_lambda"],
    unbias_domain=[
            torch.zeros(1, 1, 28, 28, device=device),
            torch.ones(1, 1, 28, 28, device=device),
        ],
    )
    l2 = L2Regulariser(lmbd=CONFIG["l2_lambda"])
    regulariser = MultiRegulariser([l2, unbias])

    if buffer_size == SMALL:
        sizes = [400, 200, 200, 200, 0]
    elif buffer_size == MEDIUM:
        sizes = [1400, 1200, 800, 600, 0]
    elif buffer_size == LARGE:
        sizes = [4800, 4000, 4000, 3200, 0]
    train_tasks, buffer_tasks = zip(
        *[create_holdout_set(dataset, holdout_size=holdout) for dataset, holdout in zip(train_tasks, sizes)]
    )
    print([len(task) for task in buffer_tasks])
    print([len(task) for task in train_tasks])

    task_labels = [torch._unique(_extract_targets(train))[0].tolist() for train in train_tasks]

    buffer = MultiTaskBuffer([])
    buffer_trainer = BufferTrainer(
        model,
        checkpoint=CONFIG["checkpoint"],
        n_iters=CONFIG["n_iters"],
        min_acc_limit=CONFIG["initial_target_acc"],
        min_acc_increment=0,
        primal_learning_rate=CONFIG["primal_learning_rate"],
        dual_learning_rate=CONFIG["dual_learning_rate"],
        projection_strategy=CONFIG["projection_strategy"],
        n_certificate_samples=CONFIG["n_certificate_samples"],
        penalty_coefficient=CONFIG["penalty_coefficient"],
        paradigm=paradigm,
        seed=SEED,
        buffer=buffer,
        domain_map_fn=domain_map_fn if paradigm == "DIL" else None,
        task_labels = task_labels
    )

    if buffer_size == SMALL:
        MAX_BUFFER_CALLS = 1
    if buffer_size == MEDIUM:
        MAX_BUFFER_CALLS = 3
    if buffer_size == LARGE:
        MAX_BUFFER_CALLS = 7
    target_acc_mapping = {
        "TIL": 0.95,
        "DIL": 0.85,
        "CIL": 0.65
    }
    target_acc = target_acc_mapping[paradigm]
    lower_bounds = []
    buffer_calls = []
    accuracy_matrix = []
    for i, (train, test, buffer) in enumerate(zip(train_tasks, test_tasks, buffer_tasks)):
        buffer_trainer.train(
            train,
            test,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            regulariser=regulariser,
            context_id=i if paradigm == "TIL" else None,
            val_freq=(len(train) // CONFIG["batch_size"]) - 1
        )
        results = buffer_trainer.test(test_tasks, context_list=list(range(len(test_tasks))) if paradigm=="TIL" else [None] * len(test_tasks))
        accs = [res[1] for res in results]
        if i == 0 and accs[0] < 0.7:
            wandb.finish(1)
            return
        # If results are not satisfactory, then use buffer data to recompute rashomon set and continue training
        j = 0
        buffer_call = 0
        prev_acc = None
        while (
            j < MAX_BUFFER_CALLS
            and results[i][1] < target_acc
            and i > 0
            and not buffer_trainer.buffer.is_empty()
        ):
            buffer_call += 1
            print_colored("Using buffer to recompute LID.", color="amber")

            buffer_trainer.recall_dataset, (buffer_X, buffer_y) = buffer_trainer.buffer.consume_merge()
            print("Recall dataset size:", len(buffer_trainer.recall_dataset))
            dataset = torch.utils.data.TensorDataset(buffer_X, buffer_y)
            buffer_trainer.compute_rashomon_set(
                dataset, use_outer_bbox=False, batch_size=len(dataset), context_id=i-1 if paradigm == "TIL" else None
            )
            buffer_trainer.train(
                train,
                test,
                batch_size=CONFIG["batch_size"],
                epochs=CONFIG["epochs"],
                lr=CONFIG["lr"],
                weight_decay=CONFIG["weight_decay"],
                regulariser=regulariser,
                early_stopping=True,
                val_freq=50,
                patience=10,
                context_id=i if paradigm == "TIL" else None
            )
            results = buffer_trainer.test(test_tasks, context_list=list(range(len(test_tasks))) if paradigm=="TIL" else [None] * len(test_tasks))
            accs = [res[1] for res in results]

            print("lower_bounds:", lower_bounds)
            diffs = [accs[i] - lower_bounds[i] for i in range(len(lower_bounds))]
            min_diff_idx = diffs.index(
                min(diffs)
            )  # The assumption is that the task closest to its boundary is the one restricting further improvements
            if results[i][1] > target_acc:
                break
            elif prev_acc is not None and results[i][1] - prev_acc < CONFIG["loosening_thresh"]:
                print("Loosening task", min_diff_idx, "bounds.")
                lower_bounds[min_diff_idx] = max(lower_bounds[min_diff_idx] - CONFIG["loosening_step"], 0.001)
                buffer_trainer.min_acc_limit = lower_bounds
            prev_acc = results[i][1]
            j += 1
        buffer_calls.append(buffer_call)

        print_colored(accs, color="green")
        accuracy_matrix.append(accs)

        lower_bounds.append(max(accs[i] - CONFIG["min_acc_increment"], 0.001))

        buffer_trainer.min_acc_limit = lower_bounds

        if i < len(train_tasks) - 1:
            buffer_trainer.compute_rashomon_set(test, context_id=i if paradigm == "TIL" else None)
            if len(buffer):
                buffer_trainer.add_to_buffer(buffer, task_id=i, k=CONFIG["buffer_k"])
        else:
            print("Buffer calls:", buffer_calls)
            accuracy_matrix.append(buffer_trainer.final_certificates + [0])
            print("final_certificates:", buffer_trainer.final_certificates)
            columns = [f"Test Task {i}" for i in range(len(test_tasks))]
            rows = [f"Task {i}" for i in range(len(test_tasks))] + ["Certificates"]
            wandb.log(
                    {
                        "accuracy_matrix": wandb.Table(
                            data=accuracy_matrix, columns=columns, rows=rows
                        )
                    }
                )

    wandb.finish(0)
if __name__ == "__main__":
    SMALL = 1000
    MEDIUM = 5000
    LARGE = 15000
    for i in range(15):
        for paradigm in ["CIL", "TIL", "DIL"]:
            for (buffer_label, buffer_size) in [('small', SMALL), ('medium', MEDIUM), ('large', LARGE)]:
                with wandb.init(project='certified-continual-learning', config=CONFIG, reinit=True, tags=["final_mnist_buffer", f"buffer_{buffer_label}", f"buffer_{paradigm.lower()}"]):
                    wandb.log({'seed': i})
                    config = wandb.config
                    run_buffer(buffer_size, i, config, paradigm=paradigm)