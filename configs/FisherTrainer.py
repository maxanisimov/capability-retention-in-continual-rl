from configs.IntervalTrainer import CONFIG as interval_config

MNIST = interval_config["MNIST"] | {
    "fisher_batch_size": 128,
    "fisher_epochs": 10,
    "prune_prop": 0.8,
    "n_iters": 400,
}

CONFIG = {"MNIST": MNIST, "CIFAR": None}
