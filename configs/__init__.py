from configs.IntervalTrainer import CONFIG as interval_trainer_config
from configs.FisherTrainer import CONFIG as fisher_trainer_config
from configs.BufferTrainer import CONFIG as buffer_trainer_config
from configs.FisherBufferTrainer import CONFIG as fisher_buffer_trainer_config
from configs.AGEMTrainer import CONFIG as agem_trainer

MNIST_IT_CONFIG = interval_trainer_config["MNIST"]
CIFAR_IT_CONFIG = interval_trainer_config["CIFAR"]
MNIST_FISHER_CONFIG = fisher_trainer_config["MNIST"]
MNIST_BUFFER_CONFIG = buffer_trainer_config["MNIST"]
MNIST_FISHER_BUFFER_CONFIG = fisher_buffer_trainer_config["MNIST"]
MNIST_AGEM_CONFIG = agem_trainer["MNIST"]