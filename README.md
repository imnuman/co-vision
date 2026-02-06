# CoVision

Computer Vision module for AI companions. Provides real-time presence detection, face recognition, gaze tracking, and scene understanding.

Designed to integrate with [Feynman](https://github.com/imnuman/feynman) voice assistant for a complete multimodal AI companion experience.

## Features

- **Presence Detection** - Real-time person detection using YOLO26
- **Face Recognition** - Identify specific users with InsightFace/ArcFace
- **Gaze Tracking** - Know when user is looking at camera via MediaPipe
- **Scene Understanding** - Describe scenes using Moondream VLM
- **Event System** - Subscribe to presence events (arrived, left, looking)
- **Feynman Integration** - Plug directly into Feynman's LiveKit agent

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CoVision Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Camera (30 FPS)                                                │
│      │                                                          │
│      ├──► YOLO26-nano ──► Person Detection (every frame)       │
│      │                                                          │
│      ├──► InsightFace ──► Face Recognition (5-10 FPS)          │
│      │                                                          │
│      ├──► MediaPipe ──► Gaze/Attention Tracking (15 FPS)       │
│      │                                                          │
│      └──► Moondream ──► Scene Understanding (on-demand)        │
│                                                                  │
│  Events: user_arrived | user_left | user_looking | scene_update │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Person Detection | YOLO26-nano | NMS-free, edge-optimized |
| Face Detection | SCRFD-500MF | ~46ms per 640x480 frame |
| Face Recognition | InsightFace (ArcFace) | 98.36% accuracy |
| Gaze Tracking | MediaPipe Face Mesh | 478 landmarks with iris |
| Scene Understanding | Moondream 2B | <2GB VRAM, runs locally |
| Camera Capture | OpenCV | Cross-platform webcam support |
| Inference Runtime | ONNX Runtime | 43% faster than OpenCV DNN |

## Installation

```bash
# Clone repository
git clone https://github.com/imnuman/co-vision.git
cd co-vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Download models (first run)
python -m covision.download_models
```

## Quick Start

### Standalone Mode

```python
from covision import VisionSystem

# Initialize vision system
vision = VisionSystem()

# Subscribe to events
@vision.on("user_arrived")
def on_user_arrived(event):
    print(f"Welcome back, {event.user_id}!")

@vision.on("user_left")
def on_user_left(event):
    print("User left the frame")

@vision.on("user_looking")
def on_looking(event):
    print("User is looking at camera - ready to listen")

# Start processing
vision.start()
```

### Feynman Integration

```python
from covision import VisionSystem
from covision.integrations.feynman import FeynmanVisionPlugin

# In your Feynman agent
vision = VisionSystem()
plugin = FeynmanVisionPlugin(vision)

# Vision context automatically injected into Mentor
agent = VoiceAgent(
    llm=FeynmanLLM(mentor, vision_context=plugin.get_context),
    ...
)
```

### User Enrollment

```bash
# Enroll your face (captures 10 reference images)
python -m covision.enroll --name "numan"

# Verify enrollment
python -m covision.verify --name "numan"
```

## Configuration

Create `configs/default.yaml`:

```yaml
camera:
  device: 0
  width: 1280
  height: 720
  fps: 30

detection:
  model: yolo26n
  confidence: 0.5
  device: cuda  # or cpu

recognition:
  model: buffalo_l
  threshold: 0.5
  embeddings_path: models/embeddings/

gaze:
  attention_threshold: 0.3
  smoothing_window: 5

scene:
  model: moondream2
  trigger: on_demand  # or periodic
  interval: 5.0  # seconds, if periodic
```

## Project Structure

```
co-vision/
├── src/
│   └── covision/
│       ├── __init__.py          # Package exports
│       ├── camera.py            # Webcam capture
│       ├── detector.py          # YOLO person detection
│       ├── recognizer.py        # Face recognition
│       ├── tracker.py           # Multi-object tracking
│       ├── attention.py         # Gaze/attention detection
│       ├── scene.py             # VLM scene understanding
│       ├── events.py            # Event emitter system
│       ├── presence.py          # High-level presence logic
│       ├── pipeline.py          # Async processing pipeline
│       └── integrations/
│           └── feynman.py       # Feynman voice integration
├── configs/
│   └── default.yaml             # Default configuration
├── models/                      # Downloaded model weights
│   └── embeddings/              # User face embeddings
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Hardware Requirements

### Minimum (CPU-only)
- CPU: 4+ cores
- RAM: 8GB
- Works but slower inference (~5 FPS detection)

### Recommended (GPU)
- GPU: NVIDIA RTX 3060+ or Apple M1/M2
- RAM: 16GB
- VRAM: 4GB+
- Achieves real-time performance (30+ FPS detection)

### Edge Deployment
- NVIDIA Jetson Orin Nano (40 TOPS)
- Raspberry Pi 5 with Coral TPU

## Performance Targets

| Task | Frequency | Latency Target |
|------|-----------|----------------|
| Frame capture | 30 FPS | <33ms |
| Person detection | 30 FPS | <10ms (GPU) |
| Face detection | 10 FPS | <50ms |
| Face recognition | 5 FPS | <100ms |
| Gaze tracking | 15 FPS | <20ms |
| Scene description | On-demand | <500ms |

## Development

```bash
# Run tests
pytest

# Run with debug logging
COVISION_DEBUG=1 python -m covision.demo

# Profile performance
python -m covision.benchmark
```

## License

MIT

## Related Projects

- [Feynman](https://github.com/imnuman/feynman) - AI voice assistant with Feynman voice clone
- [Pipecat](https://github.com/pipecat-ai/pipecat) - Voice and multimodal AI framework
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit
