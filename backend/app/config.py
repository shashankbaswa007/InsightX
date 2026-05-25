# InsightX AI - Backend Configuration

import os

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Paths
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ML Defaults
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MAX_UPLOAD_SIZE_MB = 50
