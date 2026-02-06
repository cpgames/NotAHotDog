# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NotAHotDog** - A simple PyTorch CNN image classifier that determines if an image contains a hot dog or not. Learning project focused on understanding CNN fundamentals.

## Tech Stack

- Python 3.x
- PyTorch (CNN implementation)
- torchvision (image transforms, datasets)
- PIL/Pillow (image loading)

## Build & Development Commands

```bash
# Run classifier on an image
python main.py <image_path>

# Train the model
python train.py

# Install dependencies
pip install torch torchvision pillow
```

## Project Structure

```
NotAHotDog/
├── main.py          # CLI entry point - classify an image
├── train.py         # Training script
├── model.py         # CNN architecture definition
├── data/            # Training data (hot dog / not hot dog images)
├── weights/         # Saved model weights
└── CLAUDE.md        # This file
```

## Architecture

Simple CNN with:
- 2-3 convolutional layers
- Max pooling
- Fully connected layers
- Binary classification output (hot dog vs not hot dog)

## Code Conventions

- Keep it simple - this is a learning project
- Use standard Python conventions (PEP 8)
- Clear comments explaining CNN concepts
- Explicit PyTorch code (no magic, show what's happening)
