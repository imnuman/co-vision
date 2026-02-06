# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoVision is a computer vision module for AI companions. It provides real-time presence detection, face recognition, gaze tracking, and scene understanding. Designed to integrate with [Feynman](https://github.com/imnuman/feynman) voice assistant for a complete multimodal AI companion.

## Development Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Download models (first run)
python -m covision.download_models

# Run demo
python -m covision.demo

# Run tests
pytest
pytest tests/test_detector.py -v
pytest -k "test_face"

# Benchmark performance
python -m covision.benchmark

# Enroll user face
python -m covision.enroll --name "username"
```

## Tech Stack

- **Language:** Python 3.11+
- **Person Detection:** YOLO26-nano (Ultralytics)
- **Face Detection:** SCRFD-500MF (InsightFace)
- **Face Recognition:** ArcFace via InsightFace
- **Gaze Tracking:** MediaPipe Face Mesh (478 landmarks)
- **Scene Understanding:** Moondream 2B (local VLM)
- **Camera Capture:** OpenCV
- **Inference Runtime:** ONNX Runtime (with optional TensorRT)
- **Async Framework:** asyncio + ThreadPoolExecutor

## Architecture

### Core Modules

- **covision/camera.py** - Webcam capture with async frame buffer
- **covision/detector.py** - YOLO-based person detection
- **covision/recognizer.py** - Face detection + ArcFace recognition
- **covision/attention.py** - MediaPipe gaze/attention tracking
- **covision/scene.py** - Moondream VLM scene understanding
- **covision/events.py** - Event emitter (user_arrived, user_left, user_looking)
- **covision/presence.py** - High-level state machine for presence detection
- **covision/pipeline.py** - Async pipeline orchestrating all components
- **covision/integrations/feynman.py** - Plugin for Feynman voice system

### Data Flow

```
Camera → Frame Buffer → Detection Pipeline
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    YOLO (person)      InsightFace (face)   MediaPipe (gaze)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    Presence State Machine
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
            user_arrived  user_left  user_looking
                              │
                              ▼
                    Scene Analysis (on-demand)
                              │
                              ▼
                    Feynman Mentor Context
```

### Frame Rate Strategy

| Component | Target FPS | Rationale |
|-----------|-----------|-----------|
| Camera capture | 30 | Smooth video feed |
| Person detection | 30 | Quick presence awareness |
| Face detection | 10 | Heavier, run less often |
| Face recognition | 5 | Only when new face detected |
| Gaze tracking | 15 | Needs smooth tracking |
| Scene analysis | On-demand | Too heavy for continuous |

## Key Design Decisions

- **Async pipeline:** Uses `asyncio` with `ThreadPoolExecutor` for parallel inference
- **Tiered processing:** Different models run at different frequencies to balance accuracy and performance
- **Event-driven:** Components emit events, subscribers react (decoupled from Feynman)
- **Stateful presence:** Hysteresis to avoid flickering (user must be absent N frames before "left" event)
- **Lazy loading:** Models loaded on first use, not at import
- **Device agnostic:** Auto-detects CUDA/MPS/CPU at runtime

## Configuration

Configuration in `configs/default.yaml`. Key settings:

```yaml
camera:
  device: 0              # Camera index or path
  width: 1280
  height: 720

detection:
  model: yolo26n         # yolo26n, yolov8n, yolov10n
  confidence: 0.5
  device: auto           # auto, cuda, mps, cpu

recognition:
  model: buffalo_l       # InsightFace model pack
  threshold: 0.5         # Cosine similarity threshold

gaze:
  attention_threshold: 0.3  # How centered iris must be

scene:
  model: moondream2
  trigger: on_demand     # on_demand or periodic
```

## Environment Variables

```bash
COVISION_DEBUG=1         # Enable debug logging
COVISION_DEVICE=cuda     # Force specific device
COVISION_CONFIG=path.yaml # Custom config path
```

## Feynman Integration

CoVision integrates with Feynman's `agent.py` via the plugin:

```python
# In feynman/phone/agent.py
from covision import VisionSystem
from covision.integrations.feynman import FeynmanVisionPlugin

vision = VisionSystem()
plugin = FeynmanVisionPlugin(vision, mentor)

# Plugin automatically:
# - Greets user on arrival
# - Provides scene context to Mentor.chat()
# - Triggers attention-based listening
```

## Model Downloads

Models are downloaded to `models/` directory:

| Model | Size | Downloaded By |
|-------|------|---------------|
| yolo26n.pt | ~6MB | Ultralytics auto-download |
| buffalo_l | ~300MB | `insightface` auto-download |
| mediapipe | ~2MB | `mediapipe` auto-download |
| moondream2 | ~2GB | Manual or `download_models.py` |

## Testing

```bash
# Unit tests
pytest tests/

# Integration tests (requires camera)
pytest tests/integration/ --camera

# Performance benchmarks
python -m covision.benchmark --iterations 100
```

## Common Tasks

### Add support for new YOLO version
1. Update `detector.py` to support new model variant
2. Add model name to config schema in `configs/`
3. Update benchmark script

### Tune face recognition threshold
1. Collect false positive/negative samples
2. Run `python -m covision.tune_threshold --samples ./samples/`
3. Update `configs/default.yaml` with optimal threshold

### Add new event type
1. Define event class in `events.py`
2. Emit from relevant component
3. Document in README.md
4. Add handler example in `integrations/feynman.py`
