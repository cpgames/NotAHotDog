"""
CNN Model for Hot Dog Classification

A simple Convolutional Neural Network that classifies images as
"hot dog" or "not hot dog". This is a learning project with explicit,
well-commented code to understand CNN fundamentals.
"""

import torch
import torch.nn as nn


class HotDogClassifier(nn.Module):
    """
    A simple CNN for binary image classification (hot dog vs not hot dog).

    Architecture Overview:
    - 3 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    - Sigmoid output for binary classification (0 = not hot dog, 1 = hot dog)

    Input: RGB image tensor of shape (batch_size, 3, 224, 224)
    Output: Probability tensor of shape (batch_size, 1)
    """

    def __init__(self):
        super(HotDogClassifier, self).__init__()

        # ============================================================
        # CONVOLUTIONAL LAYERS
        # These layers learn to detect features in the image
        # (edges, textures, shapes, and eventually "hot dog-ness")
        # ============================================================

        # Conv Layer 1: Detect basic features (edges, colors)
        # Input: (batch, 3, 224, 224) - 3 channels for RGB
        # Output: (batch, 32, 224, 224) - 32 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB input
            out_channels=32,    # Learn 32 different filters
            kernel_size=3,      # 3x3 filter size
            padding=1           # Padding to keep same spatial size
        )

        # Conv Layer 2: Detect more complex patterns
        # Input: (batch, 32, 112, 112) - after first pooling
        # Output: (batch, 64, 112, 112) - 64 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Conv Layer 3: Detect high-level features (hot dog shapes!)
        # Input: (batch, 64, 56, 56) - after second pooling
        # Output: (batch, 128, 56, 56) - 128 feature maps
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        # ============================================================
        # POOLING LAYER
        # Reduces spatial dimensions by half, keeping important features
        # Also provides some translation invariance
        # ============================================================

        # Max Pooling: Take the maximum value in each 2x2 region
        # This reduces image size by half in each dimension
        self.pool = nn.MaxPool2d(
            kernel_size=2,  # 2x2 pooling window
            stride=2        # Move by 2 pixels (no overlap)
        )

        # ============================================================
        # FULLY CONNECTED LAYERS
        # These layers make the final classification decision
        # ============================================================

        # After 3 conv+pool operations:
        # 224 -> 112 -> 56 -> 28 (spatial size)
        # Final feature map size: 128 channels x 28 x 28 = 100,352 values

        # FC Layer 1: Compress features into 512 neurons
        self.fc1 = nn.Linear(
            in_features=128 * 28 * 28,  # Flattened feature maps
            out_features=512
        )

        # FC Layer 2: Final classification layer
        # Output single value for binary classification
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=1  # Single output: hot dog probability
        )

        # ============================================================
        # ACTIVATION FUNCTIONS
        # ============================================================

        # ReLU: Introduces non-linearity (allows learning complex patterns)
        # ReLU(x) = max(0, x) - simple but effective
        self.relu = nn.ReLU()

        # Sigmoid: Squashes output to range [0, 1] for probability
        self.sigmoid = nn.Sigmoid()

        # Dropout: Randomly zeros some neurons during training
        # Helps prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass: Define how data flows through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
               - batch_size: number of images
               - 3: RGB channels
               - 224x224: image dimensions

        Returns:
            Tensor of shape (batch_size, 1) with values in [0, 1]
            representing probability of being a hot dog
        """
        # Input shape: (batch, 3, 224, 224)

        # ============================================================
        # CONV BLOCK 1
        # ============================================================
        x = self.conv1(x)       # -> (batch, 32, 224, 224)
        x = self.relu(x)        # Apply ReLU activation
        x = self.pool(x)        # -> (batch, 32, 112, 112)

        # ============================================================
        # CONV BLOCK 2
        # ============================================================
        x = self.conv2(x)       # -> (batch, 64, 112, 112)
        x = self.relu(x)        # Apply ReLU activation
        x = self.pool(x)        # -> (batch, 64, 56, 56)

        # ============================================================
        # CONV BLOCK 3
        # ============================================================
        x = self.conv3(x)       # -> (batch, 128, 56, 56)
        x = self.relu(x)        # Apply ReLU activation
        x = self.pool(x)        # -> (batch, 128, 28, 28)

        # ============================================================
        # FLATTEN
        # Convert 3D feature maps to 1D vector for fully connected layers
        # ============================================================
        x = x.view(x.size(0), -1)  # -> (batch, 128*28*28) = (batch, 100352)

        # ============================================================
        # FULLY CONNECTED LAYERS
        # ============================================================
        x = self.fc1(x)         # -> (batch, 512)
        x = self.relu(x)        # Apply ReLU activation
        x = self.dropout(x)     # Apply dropout (only active during training)

        x = self.fc2(x)         # -> (batch, 1)
        x = self.sigmoid(x)     # -> (batch, 1) values in [0, 1]

        return x


# Quick test to verify the model works
if __name__ == "__main__":
    # Create model instance
    model = HotDogClassifier()

    # Create a fake batch of 4 images (random noise)
    # Shape: (batch_size=4, channels=3, height=224, width=224)
    fake_images = torch.randn(4, 3, 224, 224)

    # Run forward pass
    output = model(fake_images)

    print("Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)
    print(f"\nInput shape:  {fake_images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (probabilities): {output.squeeze().tolist()}")
    print("\nModel is working correctly!")
