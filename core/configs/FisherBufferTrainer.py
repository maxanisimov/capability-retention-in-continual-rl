from configs.FisherTrainer import CONFIG as fisher_config
from configs.BufferTrainer import CONFIG as buffer_config

MNIST = (
    fisher_config["MNIST"]
    | buffer_config["MNIST"]
    | {"min_acc_increment": 0.2, "prune_prop": 0.6}
)

CONFIG = {"MNIST": MNIST, "CIFAR": None}
