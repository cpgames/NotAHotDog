"""
HotDogClassifier - CLI Entry Point

This script classifies an image as either "hot dog" or "not hot dog".

Usage:
    python main.py <image_path>

Example:
    python main.py images/food.jpg
"""

import sys
import torch
from PIL import Image
from torchvision import transforms

from model import HotDogClassifier


# Image preprocessing (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(weights_path="weights/best_model.pth"):
    """
    Load the trained HotDogClassifier model.

    Args:
        weights_path: Path to saved model weights

    Returns:
        Loaded model in evaluation mode
    """
    model = HotDogClassifier()

    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from: {weights_path}")
    except FileNotFoundError:
        print(f"Warning: No weights found at {weights_path}")
        print("Using untrained model (predictions will be random)")

    model.eval()  # Set to evaluation mode
    return model


def classify_image(model, image_path):
    """
    Classify an image as hot dog or not hot dog.

    Args:
        model: The trained HotDogClassifier
        image_path: Path to the image file

    Returns:
        Tuple of (is_hot_dog: bool, confidence: float)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()

    is_hot_dog = probability > 0.5
    confidence = probability if is_hot_dog else 1 - probability

    return is_hot_dog, confidence


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py images/food.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model
    model = load_model()

    # Classify image
    try:
        is_hot_dog, confidence = classify_image(model, image_path)
    except FileNotFoundError:
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    # Print result
    if is_hot_dog:
        print(f"\n🌭 HOT DOG! (confidence: {100 * confidence:.1f}%)")
    else:
        print(f"\n❌ NOT A HOT DOG (confidence: {100 * confidence:.1f}%)")


if __name__ == "__main__":
    main()
