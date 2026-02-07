"""
Pytest configuration and fixtures for training tests.
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import HotDogDataset, HOT_DOG_CLASS_INDEX


def pytest_addoption(parser):
    """Add custom command-line options."""
    # Default to environment variable if set, otherwise use relative path to main repo data
    # The Food101 class expects root to be parent of "food-101" directory
    default_path = os.environ.get("FOOD101_DATA_PATH", "../../data")
    parser.addoption(
        "--data-path",
        action="store",
        default=default_path,
        help="Path to data directory containing food-101 folder"
    )


@pytest.fixture(scope="session")
def data_path(request):
    """Get data path from command line or use default."""
    return request.config.getoption("--data-path")


@pytest.fixture(scope="session")
def device():
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def basic_transform():
    """Basic transform for loading images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@pytest.fixture(scope="session")
def food101_dataset(data_path, basic_transform):
    """Load Food-101 dataset for testing."""
    return datasets.Food101(
        root=data_path,
        split="train",
        download=False,  # Don't download in tests
        transform=basic_transform
    )


@pytest.fixture(scope="session")
def small_hotdog_dataset(food101_dataset):
    """
    Create a small HotDogDataset for fast testing.

    Uses only a handful of samples from each class.
    """
    # Find a small number of hot dog images and negative images
    hot_dog_indices = []
    negative_indices = []

    # Find hot dog class samples first (class 55)
    # Food101 uses _labels attribute
    labels = food101_dataset._labels

    for idx, label in enumerate(labels):
        if label == HOT_DOG_CLASS_INDEX and len(hot_dog_indices) < 10:
            hot_dog_indices.append(idx)
        elif label != HOT_DOG_CLASS_INDEX and len(negative_indices) < 10:
            negative_indices.append(idx)

        # Stop once we have enough samples
        if len(hot_dog_indices) >= 10 and len(negative_indices) >= 10:
            break

    return HotDogDataset(food101_dataset, hot_dog_indices, negative_indices)
