"""
HotDogClassifier - A simple CNN for binary classification (hot dog vs not hot dog)

This module defines the neural network architecture used for classifying images.
The network uses a standard CNN pattern: convolutional layers for feature extraction
followed by fully connected layers for classification.
"""

import torch
import torch.nn as nn


class HotDogClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for hot dog classification.

    Architecture:
    - 3 convolutional blocks (conv -> relu -> maxpool)
    - 2 fully connected layers
    - Sigmoid output for binary classification

    Input: RGB image of size 224x224 (3 x 224 x 224)
    Output: Single probability value (0 = not hot dog, 1 = hot dog)
    """

    def __init__(self):
        super(HotDogClassifier, self).__init__()

        # Convolutional layers for feature extraction
        # Each block: Conv2d -> ReLU -> MaxPool2d

        # Block 1: 3 input channels (RGB) -> 32 feature maps
        # Input: 224x224x3, Output: 112x112x32 (halved by maxpool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Block 2: 32 -> 64 feature maps
        # Input: 112x112x32, Output: 56x56x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Block 3: 64 -> 128 feature maps
        # Input: 56x56x64, Output: 28x28x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        # Flattened size: 128 * 28 * 28 = 100352

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Hidden layer
        self.fc2 = nn.Linear(512, 1)               # Output layer (single value for binary)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Squash output to [0, 1] probability

        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Tensor of shape (batch_size, 1) with probabilities
        """
        # Convolutional blocks
        # Block 1
        x = self.conv1(x)      # (batch, 3, 224, 224) -> (batch, 32, 224, 224)
        x = self.relu(x)
        x = self.pool(x)       # (batch, 32, 224, 224) -> (batch, 32, 112, 112)

        # Block 2
        x = self.conv2(x)      # (batch, 32, 112, 112) -> (batch, 64, 112, 112)
        x = self.relu(x)
        x = self.pool(x)       # (batch, 64, 112, 112) -> (batch, 64, 56, 56)

        # Block 3
        x = self.conv3(x)      # (batch, 64, 56, 56) -> (batch, 128, 56, 56)
        x = self.relu(x)
        x = self.pool(x)       # (batch, 128, 56, 56) -> (batch, 128, 28, 28)

        # Flatten for fully connected layers
        # Reshape from (batch, 128, 28, 28) to (batch, 128*28*28)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)        # (batch, 100352) -> (batch, 512)
        x = self.relu(x)
        x = self.dropout(x)    # Regularization during training

        x = self.fc2(x)        # (batch, 512) -> (batch, 1)
        x = self.sigmoid(x)    # Output probability between 0 and 1

        return x


if __name__ == "__main__":
    # Quick test to verify the model works
    model = HotDogClassifier()

    # Create a dummy input (batch of 4 RGB images, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (probabilities): {output.squeeze().tolist()}")
