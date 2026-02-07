# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NotAHotDog** - A simple PyTorch CNN image classifier that determines if an image contains a hot dog or not. Learning project focused on understanding CNN fundamentals.

## Tech Stack

- Python 3.x
- PyTorch (CNN implementation)
- torchvision (image transforms, datasets)
- PIL/Pillow (image loading)
- Flask (Slack bot server)
- ngrok (local development tunneling)

## Build & Development Commands

```bash
# Run classifier on an image
python main.py <image_path>

# Train the model
python train.py

# Install dependencies
pip install torch torchvision pillow flask slack-sdk

# Run Slack bot locally (requires ngrok)
ngrok http 3000  # In one terminal
python slack_bot.py  # In another terminal
```

## Project Structure

```
NotAHotDog/
├── main.py          # CLI entry point - classify an image
├── train.py         # Training script
├── model.py         # CNN architecture definition
├── slack_bot.py     # Slack integration server
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

## Slack Integration

The bot listens for image uploads in Slack and responds with "Hot Dog" or "Not a Hot Dog".

**Setup:**
1. Create Slack App at api.slack.com/apps
2. Enable Event Subscriptions, subscribe to `file_shared` event
3. Add Bot Token Scopes: `files:read`, `chat:write`
4. Install to workspace, copy Bot Token
5. Set `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` env vars
6. Run ngrok + slack_bot.py, set Request URL in Slack to ngrok URL
