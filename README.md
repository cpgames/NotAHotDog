# Not A Hot Dog

A CNN image classifier that determines if an image contains a hot dog or not. Inspired by the [Silicon Valley](https://www.youtube.com/watch?v=ACmydtFDTGs) app.

Uses a fine-tuned ResNet18 model trained on the Food-101 dataset.

## Usage

```bash
# Classify an image
python main.py <image_path>

# Train the model
python train.py
```

## Slack Bot

The bot watches for image uploads and responds with "Hot Dog" or "Not a Hot Dog".

### Setup

1. Create a Slack App at https://api.slack.com/apps
2. Add bot scopes: `files:read`, `chat:write`
3. Subscribe to the `file_shared` event
4. Install to workspace

### Run

```bash
# Terminal 1
ngrok http 3000

# Terminal 2
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
python slack_bot.py
```

Set the Request URL in Slack to `https://<ngrok-url>/slack/events`, then invite the bot to a channel.

## Requirements

```
torch
torchvision
pillow
flask
requests
```
