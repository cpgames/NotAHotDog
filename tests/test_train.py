"""
Tests for the training pipeline.

These tests verify:
- HotDogDataset loads images correctly
- DataLoader batching works correctly
- train_one_epoch returns valid loss and accuracy
- validate returns valid loss and accuracy
- save_checkpoint saves files correctly
- Loading weights produces consistent outputs

Usage:
    pytest tests/test_train.py --data-path /path/to/food-101 -v
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import HotDogClassifier
from train import (
    HotDogDataset,
    train_one_epoch,
    validate,
    save_checkpoint,
)


# =============================================================================
# HOTDOGDATASET TESTS
# =============================================================================

class TestHotDogDataset:
    """Tests for the HotDogDataset class."""

    def test_dataset_length(self, small_hotdog_dataset):
        """Verify dataset has expected number of samples."""
        # 10 hot dogs + 10 negatives = 20 samples
        assert len(small_hotdog_dataset) == 20

    def test_dataset_returns_tensor_and_label(self, small_hotdog_dataset):
        """Verify __getitem__ returns (tensor, label) tuple."""
        image, label = small_hotdog_dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_image_shape(self, small_hotdog_dataset):
        """Verify images have correct shape (3, 224, 224)."""
        image, _ = small_hotdog_dataset[0]

        assert image.shape == (3, 224, 224)

    def test_label_shape(self, small_hotdog_dataset):
        """Verify labels have correct shape (1,)."""
        _, label = small_hotdog_dataset[0]

        assert label.shape == (1,)

    def test_label_values(self, small_hotdog_dataset):
        """Verify labels are binary (0 or 1)."""
        # Check all samples
        for i in range(len(small_hotdog_dataset)):
            _, label = small_hotdog_dataset[i]
            assert label.item() in [0.0, 1.0]

    def test_contains_positive_samples(self, small_hotdog_dataset):
        """Verify dataset contains hot dog samples (label=1)."""
        positive_count = sum(
            1 for i in range(len(small_hotdog_dataset))
            if small_hotdog_dataset[i][1].item() == 1.0
        )
        assert positive_count == 10

    def test_contains_negative_samples(self, small_hotdog_dataset):
        """Verify dataset contains non-hot-dog samples (label=0)."""
        negative_count = sum(
            1 for i in range(len(small_hotdog_dataset))
            if small_hotdog_dataset[i][1].item() == 0.0
        )
        assert negative_count == 10


# =============================================================================
# DATALOADER TESTS
# =============================================================================

class TestDataLoader:
    """Tests for DataLoader with HotDogDataset."""

    def test_dataloader_batching(self, small_hotdog_dataset):
        """Verify DataLoader creates batches correctly."""
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)

        batch_images, batch_labels = next(iter(loader))

        assert batch_images.shape == (4, 3, 224, 224)
        assert batch_labels.shape == (4, 1)

    def test_dataloader_iteration(self, small_hotdog_dataset):
        """Verify DataLoader iterates through all data."""
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)

        total_samples = 0
        for images, labels in loader:
            total_samples += images.shape[0]

        assert total_samples == len(small_hotdog_dataset)

    def test_dataloader_shuffling(self, small_hotdog_dataset):
        """Verify DataLoader shuffling produces different orders."""
        # Create two loaders with shuffling and different seeds
        torch.manual_seed(42)
        loader1 = DataLoader(small_hotdog_dataset, batch_size=20, shuffle=True)

        torch.manual_seed(123)
        loader2 = DataLoader(small_hotdog_dataset, batch_size=20, shuffle=True)

        batch1_labels = next(iter(loader1))[1]
        batch2_labels = next(iter(loader2))[1]

        # Labels should be in different order (with high probability)
        # Note: There's a tiny chance this could fail if shuffle produces same order
        # Using different seeds makes this extremely unlikely
        assert not torch.equal(batch1_labels, batch2_labels)


# =============================================================================
# TRAIN_ONE_EPOCH TESTS
# =============================================================================

class TestTrainOneEpoch:
    """Tests for the train_one_epoch function."""

    def test_train_returns_loss_and_accuracy(self, small_hotdog_dataset, device):
        """Verify train_one_epoch returns (loss, accuracy) floats."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, accuracy = train_one_epoch(model, loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)

    def test_train_loss_positive(self, small_hotdog_dataset, device):
        """Verify training loss is positive."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)

        assert loss > 0

    def test_train_accuracy_in_range(self, small_hotdog_dataset, device):
        """Verify training accuracy is between 0 and 1."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        _, accuracy = train_one_epoch(model, loader, criterion, optimizer, device)

        assert 0 <= accuracy <= 1

    def test_train_modifies_weights(self, small_hotdog_dataset, device):
        """Verify training actually updates model weights."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial weights
        initial_weights = model.fc2.weight.data.clone()

        train_one_epoch(model, loader, criterion, optimizer, device)

        # Weights should have changed
        assert not torch.equal(initial_weights, model.fc2.weight.data)


# =============================================================================
# VALIDATE TESTS
# =============================================================================

class TestValidate:
    """Tests for the validate function."""

    def test_validate_returns_loss_and_accuracy(self, small_hotdog_dataset, device):
        """Verify validate returns (loss, accuracy) floats."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)
        criterion = nn.BCELoss()

        loss, accuracy = validate(model, loader, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)

    def test_validate_loss_positive(self, small_hotdog_dataset, device):
        """Verify validation loss is positive."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)
        criterion = nn.BCELoss()

        loss, _ = validate(model, loader, criterion, device)

        assert loss > 0

    def test_validate_accuracy_in_range(self, small_hotdog_dataset, device):
        """Verify validation accuracy is between 0 and 1."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)
        criterion = nn.BCELoss()

        _, accuracy = validate(model, loader, criterion, device)

        assert 0 <= accuracy <= 1

    def test_validate_does_not_modify_weights(self, small_hotdog_dataset, device):
        """Verify validation does not update model weights."""
        model = HotDogClassifier().to(device)
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)
        criterion = nn.BCELoss()

        # Get initial weights
        initial_weights = model.fc2.weight.data.clone()

        validate(model, loader, criterion, device)

        # Weights should not have changed
        assert torch.equal(initial_weights, model.fc2.weight.data)

    def test_validate_consistent_output(self, small_hotdog_dataset, device):
        """Verify validate produces consistent results."""
        model = HotDogClassifier().to(device)
        model.eval()  # Set to eval mode to disable dropout
        loader = DataLoader(small_hotdog_dataset, batch_size=4, shuffle=False)
        criterion = nn.BCELoss()

        loss1, acc1 = validate(model, loader, criterion, device)
        loss2, acc2 = validate(model, loader, criterion, device)

        # Results should be identical (no randomness in validation)
        assert loss1 == loss2
        assert acc1 == acc2


# =============================================================================
# SAVE_CHECKPOINT TESTS
# =============================================================================

class TestSaveCheckpoint:
    """Tests for the save_checkpoint function."""

    def test_save_creates_file(self):
        """Verify save_checkpoint creates a file."""
        model = HotDogClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model, filepath)

            assert os.path.exists(filepath)

    def test_save_file_nonzero_size(self):
        """Verify saved checkpoint file has non-zero size."""
        model = HotDogClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model, filepath)

            file_size = os.path.getsize(filepath)
            assert file_size > 0

    def test_save_creates_directories(self):
        """Verify save_checkpoint creates parent directories."""
        model = HotDogClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "deep", "nested", "weights", "model.pth")
            save_checkpoint(model, filepath)

            assert os.path.exists(filepath)

    def test_save_file_is_loadable(self):
        """Verify saved checkpoint can be loaded."""
        model = HotDogClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model, filepath)

            # Should not raise an exception
            loaded_state = torch.load(filepath, weights_only=True)
            assert isinstance(loaded_state, dict)


# =============================================================================
# LOAD WEIGHTS TESTS
# =============================================================================

class TestLoadWeights:
    """Tests for loading saved model weights."""

    def test_load_weights_same_output(self, device):
        """Verify loaded model produces same output as original."""
        model1 = HotDogClassifier().to(device)
        model1.eval()

        # Create fixed input for comparison
        torch.manual_seed(42)
        test_input = torch.randn(1, 3, 224, 224).to(device)

        # Get output before saving
        with torch.no_grad():
            output1 = model1(test_input)

        # Save and load weights
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model1, filepath)

            # Create new model and load weights
            model2 = HotDogClassifier().to(device)
            model2.load_state_dict(torch.load(filepath, weights_only=True))
            model2.eval()

            # Get output after loading
            with torch.no_grad():
                output2 = model2(test_input)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_load_weights_different_instance(self, device):
        """Verify weights can be loaded into a different model instance."""
        model1 = HotDogClassifier().to(device)
        model2 = HotDogClassifier().to(device)

        # Initially, models should have different weights (random initialization)
        # Note: With same seed they'd be the same, but we're not setting seed here

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model1, filepath)

            # Load model1's weights into model2
            model2.load_state_dict(torch.load(filepath, weights_only=True))

        # Now weights should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_load_weights_preserves_architecture(self, device):
        """Verify loaded model maintains same architecture."""
        model = HotDogClassifier().to(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "weights", "test_model.pth")
            save_checkpoint(model, filepath)

            loaded_state = torch.load(filepath, weights_only=True)

        # Check that key layers are present in state dict
        expected_keys = ["conv1.weight", "conv2.weight", "conv3.weight",
                         "fc1.weight", "fc2.weight"]
        for key in expected_keys:
            assert key in loaded_state
