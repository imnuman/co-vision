# Computer Vision System for Personal AI Companion
## Comprehensive Research Report (2025-2026)

This document provides a detailed analysis of cutting-edge approaches for building a computer vision system for a personal AI companion with real-time capabilities.

---

## Table of Contents
1. [Real-Time Presence Detection](#1-real-time-presence-detection)
2. [Face Recognition - Specific Person Identification](#2-face-recognition---specific-person-identification)
3. [Attention/Gaze Tracking](#3-attentiongaze-tracking)
4. [Scene Understanding with Local VLMs](#4-scene-understanding-with-local-vlms)
5. [Integration with Voice AI](#5-integration-with-voice-ai)
6. [Hardware Requirements & Optimization](#6-hardware-requirements--optimization)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Open Source Projects to Study](#8-open-source-projects-to-study)

---

## 1. Real-Time Presence Detection

### Latest YOLO Versions (2025-2026)

#### YOLO26 (September 2025 - Latest)
- **NMS-free inference**: Eliminates post-processing step for faster, lighter deployment
- **43% faster CPU inference** compared to previous versions
- **Multi-task unified model**: Detection, segmentation, classification, pose estimation, and oriented bounding box detection in one model
- **Size variants**: Nano (N), Small (S), Medium (M), Large (L), Extra Large (X)
- **Edge-optimized**: Specifically designed for low-power and edge devices

```python
# Example usage with Ultralytics
from ultralytics import YOLO

# Load YOLO26 nano for edge deployment
model = YOLO('yolo26n.pt')

# Real-time person detection
results = model(frame, classes=[0])  # class 0 = person
```

#### YOLOv12 (February 2025)
- State-of-the-art architecture released before YOLO26
- Strong balance of speed and accuracy

**Recommendation**: Use **YOLO26 Nano** for real-time presence detection on consumer hardware. It provides the best balance of speed and accuracy for detecting when a person enters/leaves the frame.

### Face Detection (Pre-Recognition Step)

#### SCRFD (Sample and Computation Regulation for Face Detection)
- **Anchor-free framework**: Simplified training, reduced hyperparameters
- **Processing time**: ~46ms for 640x480 images
- **Superior efficiency**: Better accuracy-efficiency tradeoff than RetinaFace

#### RetinaFace-MobileNet0.25
- **Ultra-lightweight**: 1.7MB model
- **Processing time**: ~42ms for VGA resolution on CPU
- **Real-time on CPU**: Suitable for resource-constrained environments

**Recommendation**: Use **SCRFD-500MF** for lightweight face detection, or **RetinaFace-MobileNet0.25** for CPU-only deployments.

---

## 2. Face Recognition - Specific Person Identification

### Top Libraries for Person-Specific Recognition

#### DeepFace (Recommended for Simplicity)
- **Wraps multiple backends**: VGG-Face, FaceNet, ArcFace, DeepID, Dlib, SFace, GhostFaceNet
- **Built-in real-time stream**: `DeepFace.stream()` for webcam processing
- **Easy enrollment**: Store reference images in a folder
- **Facial attributes**: Age, gender, emotion analysis included

```python
from deepface import DeepFace

# Real-time recognition with database
DeepFace.stream(
    db_path="path/to/known_faces/",
    model_name="ArcFace",
    detector_backend="retinaface"
)

# Single verification
result = DeepFace.verify(
    img1_path="frame.jpg",
    img2_path="reference/user.jpg",
    model_name="ArcFace"
)
```

#### InsightFace (Recommended for Performance)
- **State-of-the-art ArcFace**: 98.36% accuracy on MegaFace benchmark
- **512-dimensional embeddings**: Efficient comparison
- **TensorRT optimization**: 1.8x FPS boost with <0.05% accuracy drop
- **Batching benefits**: 3.2x speed-up at batch=8

```python
import insightface
from insightface.app import FaceAnalysis

# Initialize with lightweight model
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Get face embeddings
faces = app.get(frame)
for face in faces:
    embedding = face.embedding  # 512-d vector
    # Compare with stored reference embedding
```

#### face_recognition Library
- **99.38% accuracy** on Labeled Faces in the Wild benchmark
- **Built on dlib**: State-of-the-art deep learning model
- **Simple API**: Ideal for quick prototyping

### Enrollment Strategy for Specific Person
1. Capture 5-10 reference images of the target user
2. Generate embeddings using ArcFace/InsightFace
3. Store embeddings with user ID
4. Use cosine similarity threshold (typically 0.4-0.6) for matching

---

## 3. Attention/Gaze Tracking

### MediaPipe Face Mesh (Recommended)
- **468 3D facial landmarks** in real-time
- **Iris tracking**: 478 landmarks including 10 iris points
- **Cross-platform**: Works on mobile, desktop, browser
- **No dedicated depth sensor required**

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process frame
results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Extract iris positions for gaze estimation
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0]
    # Iris landmarks: 468-477 (refined mode)
    left_iris = landmarks.landmark[468:473]
    right_iris = landmarks.landmark[473:478]
```

### Gaze Direction Estimation
- **Eye Aspect Ratio (EAR)**: Blink detection
- **Iris center relative to eye corners**: Determines gaze direction (left/right/center/up/down)
- **Attention detection**: Is user looking at camera?

### Alternative Libraries

#### GazeTracking
- Python library for webcam-based eye tracking
- Provides pupil position and gaze direction
- MIT licensed

#### Deepgaze
- CNN-based head pose and gaze estimation
- Includes saliency map generation

#### WebGazer.js
- Browser-based eye tracking
- Self-calibrating model
- Useful for web-based companions

---

## 4. Scene Understanding with Local VLMs

### Small VLMs for Edge/Local Inference

#### Moondream (Recommended for Edge)
| Model | Parameters | VRAM | Best For |
|-------|-----------|------|----------|
| Moondream 0.5B | 500M | <1GB | Edge devices, Raspberry Pi |
| Moondream 2B | 2B | ~2GB | Consumer laptops, quick tasks |

**Key Features**:
- Native gaze detection capability
- Structured outputs
- ~1000 token context
- Fast inference on CPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")

# Describe scene
answer = model.answer_question(
    model.encode_image(image),
    "What is happening in this scene?",
    tokenizer
)
```

#### SmolVLM (Recommended for Resource Efficiency)
| Model | Parameters | VRAM | Notes |
|-------|-----------|------|-------|
| SmolVLM-256M | 256M | <1GB | Outperforms 300x larger Idefics-80B |
| SmolVLM-2.2B | 2.2B | 4.9GB | 59.8% avg accuracy, far less VRAM than competitors |

**Key Advantages**:
- 9x image compression via pixel shuffle
- 81 tokens per 384x384 image patch (vs 16k tokens for Qwen2-VL)
- Browser and mobile deployment possible

#### Qwen2.5-VL (Recommended for Quality)
| Model | Parameters | Context | VRAM |
|-------|-----------|---------|------|
| Qwen2.5-VL 3B | 3B | 125K tokens | Higher |
| Qwen2.5-VL 7B | 7B | 125K tokens | ~14GB |

**Key Features**:
- Best-in-class performance
- Video understanding
- Long document processing
- Agentic pipelines support

#### Florence-2 (Recommended for Multi-Task)
| Model | Parameters | Size | Notes |
|-------|-----------|------|-------|
| Florence-2-base | 230M | Small | Mobile-deployable |
| Florence-2-large | 770M | Medium | SOTA for size |

**Capabilities**:
- Captioning, detection, segmentation in one model
- Prompt-based task selection
- MIT licensed

### Running VLMs Locally

#### Ollama (Recommended for Simplicity)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run Moondream or LLaVA
ollama run moondream
ollama run llava
```

#### llama.cpp (Recommended for Performance)
- New multimodal support via `libmtmd`
- Supports: Gemma 3, SmolVLM, Pixtral, Qwen2-VL, Qwen2.5-VL
- Server mode with vision capabilities

```bash
# Run multimodal server
./llama-server -m model.gguf --mmproj mmproj.gguf --port 8080
```

---

## 5. Integration with Voice AI

### Pipecat Framework (Recommended)
Open-source Python framework for real-time voice and multimodal AI.

**Features**:
- 500-800ms round-trip latency
- Voice Activity Detection (VAD) with Silero VAD
- ASR integration (Whisper, Amazon Transcribe)
- Multimodal support (audio, video, images)

```python
from pipecat.pipeline import Pipeline
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService

# Build multimodal pipeline
pipeline = Pipeline([
    transport.input(),
    stt,              # Speech-to-text
    llm,              # Language model
    tts,              # Text-to-speech
    transport.output()
])
```

### Voice Activity Detection

#### Silero VAD (Recommended)
- <1ms per 30ms audio chunk on CPU
- 6000+ languages trained
- Works with various noise levels
- Pairs excellently with Whisper

```python
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, read_audio, _, _) = utils

# Detect speech segments
wav = read_audio('audio.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model)
```

### Speech Recognition

#### WhisperX + Silero VAD
- 3x faster than real-time
- 380-520ms end-to-end latency
- <3% word error rate
- Millisecond-level VAD decisions

```python
import whisperx

# Load model
model = whisperx.load_model("large-v3", device="cuda")

# Transcribe with VAD
audio = whisperx.load_audio("audio.wav")
result = model.transcribe(audio, batch_size=16)
```

### Vision + Voice Integration Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Event Loop                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Camera     │    │  Microphone │    │   State Manager     │  │
│  │  Pipeline   │    │  Pipeline   │    │   (User Context)    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                       │             │
│         ▼                  ▼                       │             │
│  ┌─────────────┐    ┌─────────────┐               │             │
│  │ Face Detect │    │ Silero VAD  │               │             │
│  │ + Recognize │    │             │               │             │
│  └──────┬──────┘    └──────┬──────┘               │             │
│         │                  │                       │             │
│         ▼                  ▼                       │             │
│  ┌─────────────┐    ┌─────────────┐               │             │
│  │ Gaze Track  │    │  Whisper    │               │             │
│  │ MediaPipe   │    │  STT        │               │             │
│  └──────┬──────┘    └──────┬──────┘               │             │
│         │                  │                       │             │
│         ▼                  ▼                       ▼             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Multimodal Context Fusion                    │   │
│  │   (Combine: who is present, are they looking, what said) │   │
│  └────────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LLM / VLM Processing                  │    │
│  │  (Scene understanding, response generation)              │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    TTS Output                            │    │
│  │  (ElevenLabs, Coqui, Edge TTS)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Hardware Requirements & Optimization

### Hardware Platform Comparison

| Platform | Power | Performance | Best For | Cost |
|----------|-------|-------------|----------|------|
| **Raspberry Pi 5** | 5-15W | YOLOv8n: ~5 FPS | Entry-level, simple detection | $80-100 |
| **NVIDIA Jetson Orin Nano** | 15-25W | Up to 40 TOPS | Serious CV projects | $499 |
| **NVIDIA Jetson AGX Orin** | 15-60W | Up to 275 TOPS | 4K real-time inference | $1,999 |
| **Google Coral** | 2-4W | Real-time MobileNet <50ms | Low-power quantized models | $60-150 |
| **Consumer GPU (RTX 3060+)** | 170W+ | Excellent | Desktop AI companion | $300+ |

### Optimization Techniques

#### 1. Model Quantization
- **INT8 quantization**: 75% model size reduction, minimal accuracy loss
- **FP16 precision**: 50% memory reduction on modern GPUs
- **FP8/NVFP4**: Cutting-edge for H100/newer GPUs

```python
# ONNX Runtime quantization
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8
)
```

#### 2. TensorRT Optimization
- 1.8x FPS improvement for face recognition
- FP16 conversion with <0.05% accuracy drop
- Essential for NVIDIA hardware

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

#### 3. Inference Runtime Selection

| Runtime | Best For | Speed vs OpenCV DNN |
|---------|----------|---------------------|
| **ONNX Runtime** | Cross-platform, fastest | 43% faster |
| **TensorRT** | NVIDIA GPUs | 60%+ faster |
| **OpenCV DNN** | CPU, edge devices | Baseline |
| **OpenVINO** | Intel hardware | Optimized for Intel |

#### 4. Pipeline Architecture Patterns

**Multi-threaded Pipeline**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VisionPipeline:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_frame(self, frame):
        # Run detection and recognition in parallel
        detection_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.detect_faces, frame
        )
        scene_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.analyze_scene, frame
        )

        faces, scene = await asyncio.gather(detection_task, scene_task)
        return faces, scene
```

**Producer-Consumer with Queues**:
```python
from multiprocessing import Queue, Process

# Separate processes for:
# 1. Frame capture (producer)
# 2. Detection (consumer/producer)
# 3. Recognition (consumer/producer)
# 4. VLM analysis (consumer)
```

---

## 7. Recommended Architecture

### Minimal Setup (Single Developer, Consumer Hardware)

```
Hardware: RTX 3060+ GPU or M1/M2 Mac
RAM: 16GB+
Storage: SSD for model loading

Models:
├── Person Detection: YOLO26-nano (2-3ms inference)
├── Face Detection: SCRFD-500MF or RetinaFace-MobileNet
├── Face Recognition: InsightFace/ArcFace
├── Gaze Tracking: MediaPipe Face Mesh
├── Scene Understanding: Moondream 2B or SmolVLM
├── Voice Activity: Silero VAD
└── Speech Recognition: Whisper (small/medium)
```

### Processing Pipeline

```python
class AICompanion:
    def __init__(self):
        # Initialize models
        self.yolo = YOLO('yolo26n.pt')
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.vlm = load_moondream()
        self.vad = load_silero_vad()
        self.whisper = whisperx.load_model("medium")

        # User state
        self.known_user_embedding = load_user_embedding()
        self.user_present = False
        self.user_looking = False

    async def vision_loop(self):
        while True:
            frame = await self.capture_frame()

            # Stage 1: Person detection (every frame)
            persons = self.yolo(frame, classes=[0])

            if len(persons) > 0:
                # Stage 2: Face detection + recognition (every 3-5 frames)
                faces = self.face_app.get(frame)
                for face in faces:
                    similarity = cosine_sim(face.embedding, self.known_user_embedding)
                    self.user_present = similarity > 0.5

                # Stage 3: Gaze tracking (every 2-3 frames)
                if self.user_present:
                    landmarks = self.face_mesh.process(frame)
                    self.user_looking = self.check_gaze_at_camera(landmarks)

                # Stage 4: Scene understanding (triggered, not continuous)
                if self.should_describe_scene():
                    description = self.vlm.describe(frame)
                    await self.handle_scene(description)
            else:
                self.user_present = False
                self.user_looking = False

    async def voice_loop(self):
        while True:
            audio_chunk = await self.get_audio()

            if self.vad.is_speech(audio_chunk):
                full_audio = await self.collect_speech()
                text = self.whisper.transcribe(full_audio)

                # Combine with vision context
                context = {
                    'user_present': self.user_present,
                    'user_looking': self.user_looking,
                    'text': text
                }
                await self.process_interaction(context)
```

### Frame Rate Strategy

| Task | Frequency | Rationale |
|------|-----------|-----------|
| Frame capture | 30 FPS | Smooth video |
| Person detection | 30 FPS | Quick presence awareness |
| Face detection | 10 FPS | Computationally heavier |
| Face recognition | 5 FPS | Only when new face detected |
| Gaze tracking | 15 FPS | Needs smooth tracking |
| VLM scene analysis | On-demand | Too heavy for continuous |

---

## 8. Open Source Projects to Study

### Pipecat
- **URL**: https://github.com/pipecat-ai/pipecat
- **Focus**: Voice and multimodal conversational AI
- **Key Learning**: Pipeline architecture, latency optimization

### PyGPT
- **URL**: https://pygpt.net/
- **Focus**: Desktop AI assistant with vision/voice
- **Key Learning**: GUI integration, multimodal workflow

### OpenVoiceOS
- **URL**: https://github.com/openVoiceOS
- **Focus**: Privacy-focused voice assistant
- **Key Learning**: Voice pipeline, skill system

### InsightFace
- **URL**: https://github.com/deepinsight/insightface
- **Focus**: Face analysis (detection, recognition, alignment)
- **Key Learning**: ArcFace implementation, TensorRT optimization

### VidGear
- **URL**: https://github.com/abhiTronix/vidgear
- **Focus**: High-performance video processing
- **Key Learning**: Multi-threaded capture, asyncio integration

### whisperX
- **URL**: https://github.com/m-bain/whisperX
- **Focus**: Fast Whisper with word-level timestamps
- **Key Learning**: VAD integration, batch processing

---

## Summary: Recommended Stack for Solo Developer

### Core Components

| Component | Recommended Solution | Alternative |
|-----------|---------------------|-------------|
| **Person Detection** | YOLO26-nano | YOLOv8-nano |
| **Face Detection** | SCRFD-500MF | RetinaFace-MobileNet |
| **Face Recognition** | InsightFace (ArcFace) | DeepFace |
| **Gaze Tracking** | MediaPipe Face Mesh | GazeTracking lib |
| **Scene Understanding** | Moondream 2B | SmolVLM-2.2B |
| **VLM Runtime** | Ollama or llama.cpp | Transformers |
| **Voice Activity** | Silero VAD | WebRTC VAD |
| **Speech-to-Text** | WhisperX | Faster-Whisper |
| **Framework** | Pipecat | Custom async |
| **Inference Runtime** | ONNX Runtime | OpenCV DNN |

### Quick Start Order

1. **Week 1**: Set up basic pipeline with OpenCV + YOLO for person detection
2. **Week 2**: Add face detection/recognition with InsightFace
3. **Week 3**: Integrate MediaPipe for gaze tracking
4. **Week 4**: Add Silero VAD + Whisper for voice
5. **Week 5**: Integrate Moondream for scene understanding
6. **Week 6**: Build state machine for companion logic
7. **Week 7**: Optimize with TensorRT/ONNX quantization
8. **Week 8**: Polish UX and test edge cases

---

## Sources

### Real-Time Detection
- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26: YOLO Model for Real-Time Vision AI](https://blog.roboflow.com/yolo26/)
- [Roboflow Guide to YOLO Models](https://blog.roboflow.com/guide-to-yolo-models/)

### Face Recognition
- [DeepFace GitHub](https://github.com/serengil/deepface)
- [InsightFace Project](https://www.insightface.ai/)
- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [face_recognition Library](https://github.com/ageitgey/face_recognition)

### Gaze Tracking
- [MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)
- [GazeTracking Library](https://github.com/antoinelame/GazeTracking)
- [Deepgaze Library](https://github.com/mpatacchiola/deepgaze)

### Vision Language Models
- [Moondream GitHub](https://github.com/vikhyat/moondream)
- [SmolVLM Paper](https://arxiv.org/abs/2504.05299)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [llama.cpp Multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

### Voice Integration
- [Pipecat Framework](https://github.com/pipecat-ai/pipecat)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [WhisperX](https://github.com/m-bain/whisperX)

### Optimization
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [VidGear Framework](https://github.com/abhiTronix/vidgear)

### Hardware
- [Edge AI Platform Comparison 2025](https://promwad.com/news/choose-edge-ai-platform-jetson-kria-coral-2025)
- [VLM on Edge Devices](https://learnopencv.com/vlm-on-edge-devices/)
