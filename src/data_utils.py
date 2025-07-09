import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, Subset
from typing import Tuple


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

    if not (0 < train_val_split_ratio < 1):
        raise ValueError("train_val_split_ratio must be between 0 and 1 (exclusive).")

    # Define a default transform if none is provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    full_train_dataset = datasets.EMNIST(
        root=root, split="digits", train=True, download=download, transform=transform
    )

    test_dataset = datasets.EMNIST(
        root=root, split="digits", train=False, download=download, transform=transform
    )

    total_train_size = len(full_train_dataset)
    train_size = int(train_val_split_ratio * total_train_size)
    val_size = total_train_size - train_size

    # Set random seed for reproducibility of the split
    torch.manual_seed(random_seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

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
    n_tasks=5, seed=42
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    """Get the MNIST dataset split into random tasks."""
    train_dataset, val_dataset, test_dataset = get_mnist()
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
        val_tasks.append(split_mnist_by_labels(val_dataset, labels_to_keep))
        test_tasks.append(split_mnist_by_labels(test_dataset, labels_to_keep))

    return train_tasks, val_tasks, test_tasks


def get_batch(dataset, seed=0, device="cpu", batchsize=100, domain_map_fn=None):
    """Utility function to get a batch of data from the dataset."""
    torch.manual_seed(seed)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batchsize)
    batch, labels = next(iter(dl))
    batch, labels = batch.to(device), labels.to(device)
    labels = domain_map_fn(labels) if domain_map_fn else labels
    return batch, labels


def get_context_sets(
    datasets: list[torch.utils.data.Dataset],
) -> list[list[torch.Tensor]]:
    """Get the context sets for each dataset."""
    context_sets = []
    for dataset in datasets:
        _, labels = get_batch(dataset)
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
    dataset: Dataset, holdout_fraction: float = 0.1, random_seed: int = 42
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
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    # 1. Calculate the sizes of the splits
    dataset_size = len(dataset)
    holdout_size = int(dataset_size * holdout_fraction)
    train_size = dataset_size - holdout_size

    # 2. Use a generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)

    # 3. Perform the random split
    train_dataset, holdout_dataset = random_split(
        dataset, [train_size, holdout_size], generator=generator
    )

    return train_dataset, holdout_dataset
