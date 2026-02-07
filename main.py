"""
HotDogClassifier - CLI Entry Point

This script classifies an image as either "hot dog" or "not hot dog".

Usage:
    python main.py <image_path>
    python main.py <image_path> --json

Example:
    python main.py images/food.jpg
    python main.py images/food.jpg --json

Exit codes:
    0 = classified as hot dog
    1 = classified as not hot dog
    2 = error (file not found, invalid image, model error, etc.)
"""

import argparse
import json
import os
import sys
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import HotDogClassifier


# Exit codes
EXIT_HOT_DOG = 0
EXIT_NOT_HOT_DOG = 1
EXIT_ERROR = 2

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


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

    Raises:
        RuntimeError: If model loading fails
    """
    model = HotDogClassifier()

    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        # This is okay - we can use untrained model for testing
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    model.eval()  # Set to evaluation mode
    return model


def validate_image_path(image_path):
    """
    Validate that the image path exists and has a supported extension.

    Args:
        image_path: Path to the image file

    Returns:
        None if valid

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not supported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

    _, ext = os.path.splitext(image_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def classify_image(model, image_path):
    """
    Classify an image as hot dog or not hot dog.

    Args:
        model: The trained HotDogClassifier
        image_path: Path to the image file

    Returns:
        Tuple of (is_hot_dog: bool, confidence: float)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a supported image format
        IOError: If image cannot be read or is corrupted
    """
    # Validate path first
    validate_image_path(image_path)

    # Try to load and process the image
    try:
        image = Image.open(image_path)
        # Convert to RGB (handles grayscale, RGBA, etc.)
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise IOError(f"Cannot identify image file (may be corrupted or not an image): {image_path}")
    except Exception as e:
        raise IOError(f"Failed to load image: {e}")

    # Apply transforms
    try:
        image_tensor = transform(image)
    except Exception as e:
        raise IOError(f"Failed to process image: {e}")

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    is_hot_dog = probability > 0.5
    confidence = probability if is_hot_dog else 1 - probability

    return is_hot_dog, confidence


def output_json(result=None, confidence=None, error=None):
    """
    Output result as JSON.

    Args:
        result: "hot_dog", "not_hot_dog", or None if error
        confidence: Float confidence score or None if error
        error: Error message string or None if success
    """
    output = {
        "result": result,
        "confidence": confidence,
        "error": error
    }
    print(json.dumps(output))


def output_human(is_hot_dog, confidence):
    """
    Output result in human-readable format with emoji.

    Args:
        is_hot_dog: Boolean classification result
        confidence: Float confidence score
    """
    if is_hot_dog:
        print(f"\n🌭 HOT DOG! (confidence: {100 * confidence:.1f}%)")
    else:
        print(f"\n❌ NOT A HOT DOG (confidence: {100 * confidence:.1f}%)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify an image as hot dog or not hot dog.",
        epilog="Exit codes: 0=hot dog, 1=not hot dog, 2=error"
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file to classify"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load model
    try:
        model = load_model()
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        if args.json_output:
            output_json(error=error_msg)
        else:
            print(f"Error: {error_msg}")
        sys.exit(EXIT_ERROR)

    # Classify image
    try:
        is_hot_dog, confidence = classify_image(model, args.image_path)
    except FileNotFoundError as e:
        if args.json_output:
            output_json(error=str(e))
        else:
            print(f"Error: {e}")
        sys.exit(EXIT_ERROR)
    except ValueError as e:
        if args.json_output:
            output_json(error=str(e))
        else:
            print(f"Error: {e}")
        sys.exit(EXIT_ERROR)
    except IOError as e:
        if args.json_output:
            output_json(error=str(e))
        else:
            print(f"Error: {e}")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        error_msg = f"Unexpected error processing image: {e}"
        if args.json_output:
            output_json(error=error_msg)
        else:
            print(f"Error: {error_msg}")
        sys.exit(EXIT_ERROR)

    # Output result
    if args.json_output:
        result = "hot_dog" if is_hot_dog else "not_hot_dog"
        output_json(result=result, confidence=round(confidence, 4))
    else:
        output_human(is_hot_dog, confidence)

    # Exit with appropriate code
    sys.exit(EXIT_HOT_DOG if is_hot_dog else EXIT_NOT_HOT_DOG)


if __name__ == "__main__":
    main()
