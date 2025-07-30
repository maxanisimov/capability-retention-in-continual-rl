from configs.IntervalTrainer import CONFIG as interval_config

MNIST = interval_config["MNIST"] | {
    "initial_target_acc": 0.7,
    "min_acc_increment": 0.15,
    "max_buffer_calls": 7,
    "target_acc": 0.65,
    "loosening_thresh": 0.025,
    "loosening_step": 0.01,
    "buffer_k": 200
}

CONFIG = {"MNIST": MNIST, "CIFAR": None}
