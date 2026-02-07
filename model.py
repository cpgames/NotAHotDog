"""
CNN Model for Hot Dog Classification

Uses a pretrained ResNet18 backbone with transfer learning for binary
classification (hot dog vs not hot dog). ResNet18 was trained on ImageNet
(1.2M images, 1000 classes) so it already understands visual features like
edges, textures, and shapes — we just fine-tune it for hot dogs.

This module defines the neural network architecture used for classifying images.
"""

import torch
import torch.nn as nn
from torchvision import models


class HotDogClassifier(nn.Module):
    """
    ResNet18-based classifier for hot dog detection.

    Architecture Overview:
    - Pretrained ResNet18 backbone (feature extraction)
    - Custom final layer for binary classification
    - Sigmoid output (0 = not hot dog, 1 = hot dog)

    Input: RGB image tensor of shape (batch_size, 3, 224, 224)
    Output: Probability tensor of shape (batch_size, 1)
    """

    def __init__(self):
        super(HotDogClassifier, self).__init__()

        # Load pretrained ResNet18
        # weights="IMAGENET1K_V1" loads weights trained on ImageNet
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")

        # Replace the final fully connected layer
        # Original: 512 -> 1000 (ImageNet classes)
        # Ours: 512 -> 1 (hot dog probability)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through ResNet18.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Tensor of shape (batch_size, 1) with values in [0, 1]
            representing probability of being a hot dog
        """
        return self.resnet(x)


# Quick test to verify the model works
if __name__ == "__main__":
    # Create model instance
    model = HotDogClassifier()

    # Create a fake batch of 4 images (random noise)
    fake_images = torch.randn(4, 3, 224, 224)

    # Run forward pass
    output = model(fake_images)

    print("Model Architecture: ResNet18 (pretrained)")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)
    print(f"\nInput shape:  {fake_images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (probabilities): {output.squeeze().tolist()}")
    print("\nModel is working correctly!")
