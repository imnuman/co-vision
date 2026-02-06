#!/bin/bash
# Start CoVision server on RunPod
#
# Run this on your RunPod instance after SSH-ing in:
#   wget -qO- https://raw.githubusercontent.com/imnuman/co-vision/main/scripts/start_runpod.sh | bash

set -e

echo "=========================================="
echo "  CoVision Server Setup for RunPod"
echo "=========================================="

# Clone repo if not exists
if [ ! -d "/workspace/co-vision" ]; then
    echo "Cloning CoVision repository..."
    cd /workspace
    git clone https://github.com/imnuman/co-vision.git
fi

cd /workspace/co-vision

# Update if already exists
echo "Pulling latest changes..."
git pull

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-server.txt
pip install -e .

# Create models directory
mkdir -p models/embeddings

# Start server
echo "Starting CoVision server on port 8000..."
echo ""
echo "Your server URL will be:"
echo "  wss://YOUR_RUNPOD_ID-8000.proxy.runpod.net/ws"
echo ""
echo "Open the web UI at:"
echo "  https://YOUR_RUNPOD_ID-8000.proxy.runpod.net/"
echo ""

python -m server.app
