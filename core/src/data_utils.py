import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from typing import Tuple, Callable
import os
import numpy as np


class TensorWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # For Subsets from random_split, idx will be correct
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)

    @property
    def targets(self) -> torch.Tensor:
        """Access the targets of the underlying dataset and return as a tensor."""
        # Access the .targets attribute of the original dataset
        original_targets = self.dataset.targets
        # Ensure the output is a tensor, handling both list and tensor inputs
        if not isinstance(original_targets, torch.Tensor):
            return torch.tensor(original_targets, dtype=torch.long)
        return original_targets.long()


def get_emnist_digits(
    root: str = "./data",
    train_val_split_ratio: float = 0.8,
    transform=None,
    download: bool = True,
    random_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads the EMNIST 'digits' dataset and splits it into training,
    validation, and test datasets.

    Args:
        root (str): Root directory where the dataset is stored/downloaded.
        train_val_split_ratio (float): The ratio of the original training set
                                       to be used for the training split.
                                       The rest will be for validation.
                                       (e.g., 0.8 means 80% train, 20% validation).
        transform (callable, optional): A function/transform that takes in a PIL image
                                        and returns a transformed version. Default: None.
                                        If None, a default transform will be applied.
        download (bool): If True, downloads the dataset from the internet and
                         puts it in the root directory. If dataset is already
                         downloaded, it is not downloaded again.
        random_seed (int): Seed for reproducibility of the train/validation split.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the
        (train_dataset, val_dataset, test_dataset).
    """

    # Define a default transform if none is provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    train_dataset = TensorWrapper(
        datasets.EMNIST(
            root=root,
            split="digits",
            train=True,
            download=download,
            transform=transform,
        )
    )

    test_dataset = TensorWrapper(
        datasets.EMNIST(
            root=root,
            split="digits",
            train=False,
            download=download,
            transform=transform,
        )
    )

    total_train_size = len(train_dataset)
    train_size = int(train_val_split_ratio * total_train_size)
    val_size = total_train_size - train_size

    val_dataset = None

    if train_val_split_ratio < 1:
        # Set random seed for reproducibility of the split
        torch.manual_seed(random_seed)

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_mnist(
    random_seed: int = 42, train_val_split_ratio: float = 0.8
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Get the MNIST dataset."""
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    total_train_size = len(train_dataset)
    train_size = int(train_val_split_ratio * total_train_size)
    val_size = total_train_size - train_size

    val_dataset = None

    if train_val_split_ratio < 1:
        # Set random seed for reproducibility of the split
        torch.manual_seed(random_seed)

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def _extract_targets(dataset):
    """
    Extract the targets from a dataset of various types.
    """
    if isinstance(dataset, Subset):
        base_targets = _extract_targets(dataset.dataset)
        return base_targets[dataset.indices].clone().detach()
    if hasattr(dataset, "targets"):
        return dataset.targets.clone().detach()
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset.tensors[1]
    if hasattr(dataset, "labels"):
        return torch.tensor(dataset.labels)
    if hasattr(dataset, "_labels"):
        return torch.tensor(dataset._labels)
    raise AttributeError(f"Cannot extract targets from dataset of type {type(dataset)}")


def split_mnist_by_labels(
    dataset: torch.utils.data.Dataset, labels_to_keep: list[int]
) -> torch.utils.data.ConcatDataset:
    """
    Extract a subset of the dataset based on the provided labels.
    """
    if not isinstance(dataset, torch.utils.data.ConcatDataset):
        targets = _extract_targets(dataset)
        mask = torch.isin(targets, torch.tensor(labels_to_keep))
        filtered_data = torch.utils.data.Subset(dataset, torch.where(mask)[0])
        return filtered_data
    filtered_subsets = []

    for d in dataset.datasets:  # Loop over all sub-datasets
        targets = _extract_targets(d)
        mask = torch.isin(targets, torch.tensor(labels_to_keep))
        selected_indices = torch.where(mask)[0]
        filtered_subsets.append(torch.utils.data.Subset(d, selected_indices))

    # Return a new concatenated dataset with only the filtered data
    return torch.utils.data.ConcatDataset(filtered_subsets)


def get_mnist_tasks(
    n_tasks=5, seed=42, emnist: bool = False, train_val_split_ratio: float = 0.8
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    """Get the MNIST dataset split into random tasks."""
    train_dataset, val_dataset, test_dataset = (
        get_emnist_digits(train_val_split_ratio=train_val_split_ratio)
        if emnist
        else get_mnist(train_val_split_ratio=train_val_split_ratio)
    )
    train_tasks = []
    val_tasks = []
    test_tasks = []

    torch.manual_seed(seed)

    # Randomly split the even and odd labels into subsets
    even_labels = torch.tensor([0, 2, 4, 6, 8])[torch.randperm(5)].chunk(n_tasks)
    odd_labels = torch.tensor([1, 3, 5, 7, 9])[torch.randperm(5)].chunk(n_tasks)

    # Create the tasks by combining even and odd labels and filtering the datasets
    for task_id in range(n_tasks):
        labels_to_keep = even_labels[task_id].tolist() + odd_labels[task_id].tolist()
        train_tasks.append(split_mnist_by_labels(train_dataset, labels_to_keep))
        if val_dataset is not None:
            val_tasks.append(split_mnist_by_labels(val_dataset, labels_to_keep))
        test_tasks.append(split_mnist_by_labels(test_dataset, labels_to_keep))

    return train_tasks, val_tasks, test_tasks


def get_context_sets(
    datasets: list[torch.utils.data.Dataset],
) -> list[list[torch.Tensor]]:
    """Get the context sets for each dataset."""
    context_sets = []
    for dataset in datasets:
        _, labels = get_batch(dataset, batch_size=100)
        context_sets.append(labels.unique().tolist())
    return context_sets


class TaskAugmentedDataset(Dataset):
    def __init__(self, base_dataset, task_id):
        self.base_dataset = base_dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return (image, self.task_id), label - self.task_id * 2


class BinaryMNIST(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return img, self.label_map[int(label)]


def create_holdout_set(
    dataset: Dataset, holdout_size: int = 400, random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Splits a PyTorch Dataset into a training set and a random holdout set.

    Args:
        dataset (Dataset): The full dataset to be split.
        holdout_fraction (float): The fraction of the dataset to be used as the holdout set.
                                  Must be between 0 and 1. Defaults to 0.1.
        random_seed (int): A seed for the random number generator to ensure reproducibility.
                           Defaults to 42.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training dataset and the holdout dataset.
    """
    # 1. Calculate the sizes of the splits
    dataset_size = len(dataset)
    train_size = dataset_size - holdout_size

    # 2. Use a generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)

    # 3. Perform the random split
    train_dataset, holdout_dataset = random_split(
        dataset, [train_size, holdout_size], generator=generator
    )

    return train_dataset, holdout_dataset


def count_labels(labels_tensor: torch.Tensor) -> tuple[list[int], list[int]]:
    """
    Counts the occurrences of each label in a torch tensor.

    Args:
      labels_tensor: A torch tensor of labels.

    Returns:
      A tuple containing two tensors: (unique_labels, counts).
    """
    values, counts = torch.unique(labels_tensor, return_counts=True)
    return values.tolist(), counts.tolist()


def get_batch(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
    x, y = next(iter(loader))
    return x.to(device), y.to(device)

def balance_dataset_by_duplication(dataset):
    """
    Given a PyTorch dataset with binary labels (0 and 1),
    returns a new dataset where class 1 is duplicated until classes are balanced.
    """
    # Step 1: Collect indices by class
    class_0_indices = []
    class_1_indices = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == 0:
            class_0_indices.append(i)
        elif label == 1:
            class_1_indices.append(i)
        else:
            print("Dataset contains labels other than 0 or 1, won't balance.")
            return dataset

    # Step 2: Duplicate class 1 to match class 0
    n_0 = len(class_0_indices)
    n_1 = len(class_1_indices)
    if n_1 > n_0:
        temp = class_0_indices  # swap balance
        class_0_indices = class_1_indices
        class_1_indices = temp
        temp = n_0
        n_0 = n_1
        n_1 = temp

    if n_1 == 0:
        raise ValueError("No class 1 samples found — cannot balance.")

    multiplier = n_0 // n_1
    remainder = n_0 % n_1

    duplicated_1_indices = class_1_indices * multiplier + class_1_indices[:remainder]

    # Step 3: Combine and shuffle
    balanced_indices = class_0_indices + duplicated_1_indices
    balanced_indices = torch.tensor(balanced_indices)[torch.randperm(len(balanced_indices))]

    return Subset(dataset, balanced_indices)

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, feature_path: str, label_path: str, mmap=True):
        self._features = np.load(feature_path, mmap_mode="r" if mmap else None)
        self._labels = np.load(label_path)

    def __len__(self):
        return self._labels.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self._features[idx]), torch.tensor(self._labels[idx]).long()

class EmbeddingDatasetExtractor:
    def __init__(self, base_path: str, label_map_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        self._base_path = base_path
        self._dataset_cache = {}
        self._label_map_fn = label_map_fn

    def get_embedding_dataset(
        self, model_name: str, dataset_name: str, val_ratio: float = 0.1, seed=42, balance=True, use_cache=True
    ):
        result_data = None
        if model_name not in self._dataset_cache:
            self._dataset_cache[model_name] = {}
        if dataset_name not in self._dataset_cache[model_name] or not use_cache:
            print("Dataset not found or cache not used, extracting it now.")

            dataset = NpyDataset(
                os.path.join(self._base_path, f"{dataset_name}_{model_name}_features.npy"),
                os.path.join(self._base_path, f"{dataset_name}_{model_name}_labels.npy"),
            )
            if self._label_map_fn is not None:
                print("Applying map function to labels...")
                dataset._labels = self._label_map_fn(dataset._labels)
            dataset_size = len(dataset)
            val_size = int(dataset_size * val_ratio)

            # Deterministic shuffling
            generator = torch.Generator().manual_seed(seed)
            shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()

            val_indices = shuffled_indices[:val_size]
            train_indices = shuffled_indices[val_size:]

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            result_data = (
                (train_dataset, val_dataset),  # non-balanced version
                (balance_dataset_by_duplication(train_dataset), balance_dataset_by_duplication(val_dataset)),
            )
            if use_cache:
                self._dataset_cache[model_name][dataset_name] = result_data
        else:
            result_data = self._dataset_cache[model_name][dataset_name]  # using cache here

        if balance:
            return result_data[1]
        else:
            return result_data[0]