MNIST = {
    "SITrainer": {
        "CIL": {
            "method": "random",
            "metric": {"name": "val_acc", "goal": "maximize"},
            "parameters": {
                "train.l2_lambda": {"distribution": "uniform", "min": 0, "max": 1},
                "train.unbias_lambda": {"distribution": "uniform", "min": 0, "max": 1},
                "init.seed": {"values": list(range(0, 1000))},
                "init.n_certificate_samples": {"values": [128, 256, 512]},
                "init.min_acc_limit": {
                    "distribution": "uniform",
                    "min": 0.7,
                    "max": 0.95,
                },
                "init.min_acc_increment": {"values": [0.05, 0.1, 0.25]},
                "init.paradigm": {"value": "CIL"},
                "init.n_iters": {"values": [50, 100, 200, 400]},
                "init.primal_learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 0.75,
                },
                "init.dual_learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 0.75,
                },
                "init.penalty_coefficient": {
                    "distribution": "uniform",
                    "min": 1e-1,
                    "max": 10,
                },
                "init.checkpoint": {"values": [10, 20, 50, 100]},
                "init.projection_strategy": {"values": ["closest", "sample_largest_closest"]},
                "train.si_batch_size": {"values": [32, 64, 128, 256]},
                "train.si_steps": {"values": [1, 3, 5, 10, 25, 50, 100]},
                "train.prune_prop": {
                    "distribution": "uniform",
                    "min": 1e-2,
                    "max": 0.99,
                },
                "train.lr": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 5e-1,
                },
                "train.weight_decay": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 5e-1,
                },
                "simple_train.epochs": {"values": [3, 5, 10]},
                "simple_train.batch_size": {"values": [32, 64, 128, 256, 512]},
                "simple_train.lr": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 1e-1,
                },
                "simple_train.weight_decay": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 5e-1,
                },
            },
        },
    },
    "IntervalTrainer": {
        "CIL": {
            "method": "bayes",
            "metric": {"name": "final_total_accuracy", "goal": "maximize"},
            "parameters": {
                "init.projection_strategy": {"values": ["closest", "sample_largest_closest"]},
                "init.seed": {"values": list(range(5))},
                "init.n_certificate_samples": {"value": 400},
                "init.min_acc_limit": {
                    "distribution": "uniform",
                    "min": 0.8,
                    "max": 1,
                },
                "init.min_acc_increment": {
                    "distribution": "uniform",
                    "min": 0.05,
                    "max": 0.25,
                },
                "init.paradigm": {"value": "CIL"},
                "init.n_iters": {"values": [50, 100, 200, 400]},
                "init.primal_learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-2,
                    "max": 0.8,
                },
                "init.dual_learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 0.5,
                },
                "init.penalty_coefficient": {
                    "distribution": "uniform",
                    "min": 1e-1,
                    "max": 10,
                },
                "init.checkpoint": {"values": [10, 20, 50, 100]},
                "train.l2_lambda": {"distribution": "uniform", "min": 0, "max": 3e-1},
                "train.unbias_lambda": {"distribution": "uniform", "min": 0, "max": 3e-1},
                "train.lr": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 2e-1,
                },
                "train.weight_decay": {
                    "distribution": "uniform",
                    "min": 0,
                    "max": 1e-1,
                },
                "train.epochs": {"values": [3, 5, 10]},
                "train.batch_size": {"values": [32, 64, 128, 256]},
                "simple_train.epochs": {"values": [3, 5, 10]},
                "simple_train.batch_size": {"values": [32, 64, 128, 256]},
                "simple_train.lr": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 2e-1,
                },
                "simple_train.weight_decay": {
                    "distribution": "uniform",
                    "min": 0,
                    "max": 1e-1,
                },
            },
        },
    }
}

SWEEP_CONFIGS = {"MNIST": MNIST}
