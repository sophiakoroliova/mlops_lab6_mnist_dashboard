"""
src/data_loader.py
MNIST dataset loading utilities.
No model inference here — strictly data access.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CLASS_NAMES = [str(i) for i in range(10)]


@dataclass
class SplitInfo:
    train_size: int
    val_size: int
    test_size: int
    num_classes: int
    class_names: list[str]


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def load_datasets(
    data_path: str,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> tuple[Subset, Subset, datasets.MNIST]:
    """
    Download (if needed) and return (train, val, test) MNIST datasets.
    """
    logger.info("Loading MNIST from %s", data_path)
    transform = get_transform()

    full_train = datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        train_size, val_size, len(test_ds),
    )
    return train_ds, val_ds, test_ds


def get_split_info(
    train_ds: Subset,
    val_ds: Subset,
    test_ds: datasets.MNIST,
) -> SplitInfo:
    return SplitInfo(
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        num_classes=10,
        class_names=CLASS_NAMES,
    )


def get_class_distribution(dataset) -> dict[str, int]:
    """Return {class_name: count} for any dataset supporting integer labels."""
    counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for _, label in dataset:
        counts[CLASS_NAMES[int(label)]] += 1
    return counts


def get_sample(dataset, index: int) -> tuple[torch.Tensor, int]:
    """Return (image_tensor, label) for a given index."""
    image, label = dataset[index]
    return image, int(label)


def get_samples_by_class(dataset, class_idx: int) -> list[int]:
    """Return all dataset indices whose label == class_idx."""
    indices = []
    for i, (_, label) in enumerate(dataset):
        if int(label) == class_idx:
            indices.append(i)
    return indices


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised (C,H,W) tensor back to a displayable (H,W) uint8 array.
    """
    img = image.squeeze().numpy()
    img = img * MNIST_STD[0] + MNIST_MEAN[0]   # de-normalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def get_dataloader(dataset, batch_size: int = 256, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
