"""
Training Pipeline for HotDogClassifier

This script handles the complete training process:
1. Loading and preprocessing the Food-101 dataset
2. Running the training loop with validation
3. Saving model checkpoints

Usage:
    python train.py

The script will:
- Download Food-101 dataset if not present (to ./data)
- Train for specified epochs with progress bars
- Save best model (lowest val loss) and final model to ./weights
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import HotDogClassifier


# =============================================================================
# HYPERPARAMETERS (hardcoded with sensible defaults for learning)
# =============================================================================

EPOCHS = 10          # Number of complete passes through the dataset
BATCH_SIZE = 32      # Number of images processed before updating weights
LEARNING_RATE = 0.001  # Step size for optimizer (Adam default)
VAL_SPLIT = 0.2      # 20% of data for validation
NUM_WORKERS = 4      # Parallel data loading workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hot dog is class 55 in Food-101 (0-indexed alphabetically)
HOT_DOG_CLASS_INDEX = 55

# How many negative samples to use (for balanced dataset)
# Food-101 has ~1000 images per class, we'll sample similar amount of negatives
NEGATIVE_SAMPLES_PER_CLASS = 100  # From each non-hot-dog class


# =============================================================================
# DATASET HANDLING
# =============================================================================

class HotDogDataset(Dataset):
    """
    Custom dataset that wraps Food-101 and creates binary labels.

    Positive class (1): hot_dog images
    Negative class (0): randomly sampled images from other food classes
    """

    def __init__(self, food101_dataset, hot_dog_indices, negative_indices):
        """
        Args:
            food101_dataset: The base Food-101 dataset
            hot_dog_indices: Indices of hot dog images (positive samples)
            negative_indices: Indices of non-hot dog images (negative samples)
        """
        self.dataset = food101_dataset

        # Combine positive and negative samples
        # Store tuples of (original_index, binary_label)
        self.samples = []

        # Add hot dog images with label 1
        for idx in hot_dog_indices:
            self.samples.append((idx, 1.0))

        # Add non-hot dog images with label 0
        for idx in negative_indices:
            self.samples.append((idx, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx, label = self.samples[idx]
        image, _ = self.dataset[original_idx]  # Ignore original Food-101 label
        return image, torch.tensor([label], dtype=torch.float32)


def get_data_loaders():
    """
    Create training and validation data loaders.

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    print("Setting up data loaders...")

    # Define image transformations
    # Training: includes data augmentation to improve generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),           # Slightly larger for random crop
        transforms.RandomCrop(224),              # Random 224x224 crop
        transforms.RandomHorizontalFlip(),       # 50% chance to flip
        transforms.RandomRotation(10),           # Small rotation for variety
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
        transforms.ToTensor(),                   # Convert to tensor (0-1 range)
        transforms.Normalize(                    # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation: no augmentation, just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Download/load Food-101 dataset
    # Note: First run will download ~5GB of data
    print("Loading Food-101 dataset (may download on first run)...")

    # We need to load it twice with different transforms
    # Or we can apply transforms in our custom dataset
    # For simplicity, we'll load with train transform and handle val separately
    food101_train = datasets.Food101(
        root="./data",
        split="train",
        download=True,
        transform=train_transform
    )

    food101_val = datasets.Food101(
        root="./data",
        split="train",  # We'll split training data ourselves
        download=True,
        transform=val_transform
    )

    print(f"Food-101 loaded. Total training images: {len(food101_train)}")

    # Find hot dog images and sample negative images
    print("Filtering hot dog images and sampling negatives...")

    hot_dog_indices = []
    other_class_indices = {i: [] for i in range(101) if i != HOT_DOG_CLASS_INDEX}

    # Scan through dataset to categorize by class
    # Note: This is slow but only done once
    for idx in tqdm(range(len(food101_train)), desc="Scanning dataset"):
        _, label = food101_train.samples[idx]  # Get label without loading image
        if label == HOT_DOG_CLASS_INDEX:
            hot_dog_indices.append(idx)
        else:
            other_class_indices[label].append(idx)

    print(f"Found {len(hot_dog_indices)} hot dog images")

    # Sample negative images from each other class
    import random
    random.seed(42)  # For reproducibility

    negative_indices = []
    for class_idx, indices in other_class_indices.items():
        # Sample up to NEGATIVE_SAMPLES_PER_CLASS from each class
        n_samples = min(NEGATIVE_SAMPLES_PER_CLASS, len(indices))
        negative_indices.extend(random.sample(indices, n_samples))

    print(f"Sampled {len(negative_indices)} negative images from {len(other_class_indices)} classes")

    # Create train/val split
    # Shuffle and split both positive and negative samples
    random.shuffle(hot_dog_indices)
    random.shuffle(negative_indices)

    n_hot_dog_val = int(len(hot_dog_indices) * VAL_SPLIT)
    n_negative_val = int(len(negative_indices) * VAL_SPLIT)

    train_hot_dog = hot_dog_indices[n_hot_dog_val:]
    val_hot_dog = hot_dog_indices[:n_hot_dog_val]

    train_negative = negative_indices[n_negative_val:]
    val_negative = negative_indices[:n_negative_val]

    print(f"Training set: {len(train_hot_dog)} hot dogs, {len(train_negative)} negatives")
    print(f"Validation set: {len(val_hot_dog)} hot dogs, {len(val_negative)} negatives")

    # Create custom datasets
    train_dataset = HotDogDataset(food101_train, train_hot_dog, train_negative)
    val_dataset = HotDogDataset(food101_val, val_hot_dog, val_negative)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    return train_loader, val_loader


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (BCE)
        optimizer: Optimizer (Adam)
        device: Device to train on (CPU/GPU)

    Returns:
        avg_loss: Average training loss for the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()  # Set model to training mode (enables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for batches
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        # Move data to device (GPU if available)
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients from previous batch
        # (PyTorch accumulates gradients by default)
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights using gradients
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * images.size(0)

        # Calculate accuracy (threshold at 0.5)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.1f}%"
        })

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.

    Args:
        model: The neural network
        val_loader: DataLoader for validation data
        criterion: Loss function (BCE)
        device: Device to run on (CPU/GPU)

    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients needed for validation (saves memory and computation)
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_checkpoint(model, filepath, message=None):
    """
    Save model weights to file.

    Args:
        model: The neural network
        filepath: Path to save weights
        message: Optional message to print
    """
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model state dict (just the weights, not the architecture)
    torch.save(model.state_dict(), filepath)

    if message:
        print(message)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    """Main training function."""
    print("=" * 60)
    print("HotDogClassifier Training Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60)

    # Get data loaders
    train_loader, val_loader = get_data_loaders()

    # Initialize model
    print("\nInitializing model...")
    model = HotDogClassifier()
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function: Binary Cross Entropy
    # Good for binary classification where model outputs probabilities
    criterion = nn.BCELoss()

    # Optimizer: Adam (adaptive learning rate, works well in most cases)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training tracking
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {100 * train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {100 * val_acc:.2f}%")

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(
                model,
                "weights/best_model.pth",
                f">>> New best model saved! (val_loss: {val_loss:.4f})"
            )

    # Save final model
    save_checkpoint(
        model,
        "weights/final_model.pth",
        "\nFinal model saved to weights/final_model.pth"
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {100 * best_val_acc:.2f}%")
    print(f"\nModel weights saved to:")
    print(f"  - weights/best_model.pth (best validation loss)")
    print(f"  - weights/final_model.pth (final epoch)")


if __name__ == "__main__":
    main()
