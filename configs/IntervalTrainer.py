MNIST = {
    "projection_strategy": "sample_largest_closest",
    "n_certificate_samples": 400,
    "min_acc_limit": 1,
    "min_acc_increment": 0.12,
    "n_iters": 200,
    "primal_learning_rate": 0.33,
    "dual_learning_rate": 0.01,
    "penalty_coefficient": 1,
    "checkpoint": 20,
    "l2_lambda": 0.01,
    "unbias_lambda": 0.01,
    "lr": 0.001,
    "weight_decay": 0,
    "epochs": 5,
    "batch_size": 64,
}

CIFAR = MNIST | {
    "projection_strategy": "best_loss",
    "min_acc_increment": 0.2,
    "lr": 0.02,
    "batch_size": 128
}

CONFIG = {"MNIST": MNIST, "CIFAR": CIFAR}
