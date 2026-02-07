"""
Slack Bot for Hot Dog Classification

A Flask server that handles Slack events and classifies uploaded images
as "Hot Dog" or "Not a Hot Dog".

Setup:
1. Create Slack App at api.slack.com/apps
2. Enable Event Subscriptions, subscribe to `file_shared` event
3. Add Bot Token Scopes: `files:read`, `chat:write`
4. Install to workspace, copy Bot Token
5. Set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET env vars
6. Run ngrok + this script, set Request URL in Slack to ngrok URL

Usage:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_SIGNING_SECRET=...
    python slack_bot.py
"""

import hashlib
import hmac
import io
import os
import time

import requests
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from model import HotDogClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)

# Slack credentials from environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")

# Model setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weights/best_model.pth"

# Image preprocessing (same as validation transforms in train.py)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load the trained HotDogClassifier model."""
    model = HotDogClassifier()

    # Try to load weights if they exist
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model = model.to(DEVICE)
    model.eval()
    return model

# Load model at startup
MODEL = load_model()

# =============================================================================
# SLACK UTILITIES
# =============================================================================

def verify_slack_signature(request_body, timestamp, signature):
    """
    Verify that the request came from Slack using the signing secret.

    Args:
        request_body: Raw request body bytes
        timestamp: X-Slack-Request-Timestamp header
        signature: X-Slack-Signature header

    Returns:
        True if signature is valid, False otherwise
    """
    if not SLACK_SIGNING_SECRET:
        return False

    # Check timestamp to prevent replay attacks (5 minute window)
    try:
        if abs(time.time() - float(timestamp)) > 300:
            return False
    except (ValueError, TypeError):
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    expected_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_sig, signature)


def is_bot_in_channel(channel_id):
    """
    Check if the bot is a member of the specified channel.

    Args:
        channel_id: Slack channel ID

    Returns:
        True if bot is in channel, False otherwise
    """
    try:
        response = requests.post(
            "https://slack.com/api/conversations.info",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            data={"channel": channel_id}
        )
        data = response.json()

        if not data.get("ok"):
            return False

        # Check if bot is a member
        return data.get("channel", {}).get("is_member", False)
    except Exception:
        return False


def get_file_info(file_id):
    """
    Get file information from Slack.

    Args:
        file_id: Slack file ID

    Returns:
        File info dict or None if error
    """
    try:
        response = requests.get(
            "https://slack.com/api/files.info",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            params={"file": file_id}
        )
        data = response.json()

        if data.get("ok"):
            return data.get("file")
        return None
    except Exception:
        return None


def download_image(url):
    """
    Download an image from Slack's servers.

    Args:
        url: Private download URL for the file

    Returns:
        PIL Image or None if error
    """
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        )

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        return None
    except Exception:
        return None


def send_message(channel_id, text):
    """
    Send a message to a Slack channel.

    Args:
        channel_id: Slack channel ID
        text: Message text
    """
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": channel_id, "text": text}
        )
    except Exception:
        pass  # Silent mode - ignore errors


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_image(image):
    """
    Classify an image as hot dog or not.

    Args:
        image: PIL Image

    Returns:
        True if hot dog, False otherwise
    """
    # Preprocess the image
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Run inference
    with torch.no_grad():
        output = MODEL(tensor)
        probability = output.item()

    # Threshold at 0.5
    return probability > 0.5


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack event webhooks."""
    # Get raw body for signature verification
    body = request.get_data()

    # Verify request signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(body, timestamp, signature):
        return jsonify({"error": "invalid signature"}), 401

    # Parse JSON body
    data = request.get_json()

    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge")})

    # Handle events
    if data.get("type") == "event_callback":
        event = data.get("event", {})

        # Handle file_shared event
        if event.get("type") == "file_shared":
            handle_file_shared(event)

    # Always return 200 OK quickly to acknowledge receipt
    return jsonify({"ok": True})


def handle_file_shared(event):
    """
    Handle a file_shared event.

    Only processes the first image, only responds in channels where
    the bot is invited, and operates silently (no error messages).

    Args:
        event: Slack event dict
    """
    file_id = event.get("file_id")
    channel_id = event.get("channel_id")

    if not file_id or not channel_id:
        return  # Silent

    # Check if bot is in the channel
    if not is_bot_in_channel(channel_id):
        return  # Silent - not invited to this channel

    # Get file info
    file_info = get_file_info(file_id)
    if not file_info:
        return  # Silent

    # Check if it's an image
    mimetype = file_info.get("mimetype", "")
    if not mimetype.startswith("image/"):
        return  # Silent - not an image

    # Get download URL (use url_private_download for best quality)
    download_url = file_info.get("url_private_download") or file_info.get("url_private")
    if not download_url:
        return  # Silent

    # Download the image
    image = download_image(download_url)
    if not image:
        return  # Silent

    # Classify the image
    is_hot_dog = classify_image(image)

    # Send result (only output - classification result)
    if is_hot_dog:
        send_message(channel_id, "Hot Dog 🌭")
    else:
        send_message(channel_id, "Not a Hot Dog")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Hot Dog Classifier Slack Bot...")
    print(f"Device: {DEVICE}")
    print(f"Model loaded: {os.path.exists(MODEL_PATH)}")
    print("Listening on port 3000")
    print("\nRemember to:")
    print("  1. Set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET env vars")
    print("  2. Run ngrok: ngrok http 3000")
    print("  3. Set Request URL in Slack to: <ngrok-url>/slack/events")

    app.run(host="0.0.0.0", port=3000, debug=False)
