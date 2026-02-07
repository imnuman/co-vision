"""CoVision WebSocket Server for RunPod.

This server receives video frames from browsers (phone/desktop) and runs
vision inference on GPU, returning detection results in real-time.

Run with: python -m server.app
Or with uvicorn: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# Import CoVision components
from covision.detector import PersonDetector
from covision.recognizer import FaceRecognizer
from covision.attention import AttentionTracker
from covision.scene import SceneAnalyzer
from covision.presence import PresenceManager
from covision.events import EventEmitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CoVision Server")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
detector: Optional[PersonDetector] = None
recognizer: Optional[FaceRecognizer] = None
attention: Optional[AttentionTracker] = None
scene_analyzer: Optional[SceneAnalyzer] = None


class SessionState:
    """Per-connection session state."""

    def __init__(self):
        self.events = EventEmitter()
        self.presence = PresenceManager(events=self.events)
        self.frame_count = 0
        self.pending_events: list[dict] = []

        # Register event handlers
        self.events.on("user_arrived", self._on_arrived)
        self.events.on("user_left", self._on_left)
        self.events.on("user_looking", self._on_looking)

    def _on_arrived(self, event):
        self.pending_events.append({
            "type": "event",
            "event": "user_arrived",
            "message": f"User arrived (confidence: {event.confidence:.0%})",
        })

    def _on_left(self, event):
        self.pending_events.append({
            "type": "event",
            "event": "user_left",
            "message": f"User left (was here for {event.duration_seconds:.1f}s)",
        })

    def _on_looking(self, event):
        self.pending_events.append({
            "type": "event",
            "event": "user_looking",
            "message": "User is looking at camera",
        })

    def get_pending_events(self) -> list[dict]:
        events = self.pending_events
        self.pending_events = []
        return events


@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    global detector, recognizer, attention, scene_analyzer

    logger.info("Initializing CoVision models...")

    # Object detector (detects all COCO classes, not just persons)
    detector = PersonDetector(
        model=os.getenv("YOLO_MODEL", "yolov8n"),
        confidence=0.5,
        device="auto",
        classes=None,  # None = detect all 80 COCO classes
    )
    detector.load()
    detector.warmup()
    logger.info("Person detector loaded")

    # Face recognizer
    recognizer = FaceRecognizer(
        model=os.getenv("FACE_MODEL", "buffalo_l"),
        threshold=float(os.getenv("FACE_THRESHOLD", "0.5")),
        embeddings_path=os.getenv("EMBEDDINGS_PATH", "models/embeddings"),
    )
    recognizer.load()
    recognizer.warmup()
    logger.info("Face recognizer loaded")

    # Attention tracker
    attention = AttentionTracker(
        attention_threshold=0.3,
    )
    attention.load()
    attention.warmup()
    logger.info("Attention tracker loaded")

    # Scene analyzer (optional, heavy)
    if os.getenv("ENABLE_SCENE", "true").lower() == "true":
        scene_analyzer = SceneAnalyzer(
            model=os.getenv("VLM_MODEL", "moondream2"),
            device="auto",
        )
        # Don't load until first use (it's heavy)
        logger.info("Scene analyzer ready (lazy load)")

    logger.info("All models initialized!")


@app.get("/")
async def index():
    """Serve the web UI."""
    return FileResponse("web/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "detector": detector is not None,
        "recognizer": recognizer is not None,
        "attention": attention is not None,
        "scene": scene_analyzer is not None,
    }


class EnrollmentRequest(BaseModel):
    """Request body for enrollment."""
    name: str
    user_id: Optional[str] = None
    frames: List[str]  # Base64-encoded JPEG images


@app.post("/enroll")
async def enroll_user(request: EnrollmentRequest):
    """Enroll a user's face for recognition.

    Receives multiple face images and creates embeddings for recognition.
    """
    global recognizer

    if recognizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Face recognizer not initialized"}
        )

    name = request.name.strip()
    user_id = request.user_id or name.lower().replace(" ", "_")

    if not name:
        return JSONResponse(
            status_code=400,
            content={"error": "Name is required"}
        )

    if len(request.frames) < 3:
        return JSONResponse(
            status_code=400,
            content={"error": "At least 3 face images required"}
        )

    # Decode frames
    frames = []
    for i, frame_b64 in enumerate(request.frames):
        try:
            # Remove data URL prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]

            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                frames.append(frame)
        except Exception as e:
            logger.warning(f"Failed to decode frame {i}: {e}")

    if len(frames) < 3:
        return JSONResponse(
            status_code=400,
            content={"error": f"Only {len(frames)} valid frames decoded, need at least 3"}
        )

    logger.info(f"Enrolling user '{name}' (ID: {user_id}) with {len(frames)} frames")

    # Run enrollment in thread pool
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(
        None,
        recognizer.enroll,
        user_id,
        name,
        frames,
    )

    if success:
        logger.info(f"Successfully enrolled user '{name}'")
        return {
            "status": "success",
            "user_id": user_id,
            "name": name,
            "frames_used": len(frames),
        }
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Enrollment failed - could not detect face in images"}
        )


@app.get("/users")
async def list_users():
    """List enrolled users."""
    global recognizer

    if recognizer is None:
        return {"users": []}

    users = []
    for user_id, profile in recognizer._users.items():
        users.append({
            "user_id": user_id,
            "name": profile.name,
        })

    return {"users": users}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing."""
    await websocket.accept()
    logger.info("Client connected")

    session = SessionState()

    try:
        while True:
            # Receive data (can be binary frame or JSON command)
            data = await websocket.receive()

            if "bytes" in data:
                # Binary frame data
                result = await process_frame(data["bytes"], session)

                # Send detection result
                await websocket.send_json(result)

                # Send any pending events
                for event in session.get_pending_events():
                    await websocket.send_json(event)

            elif "text" in data:
                # JSON command
                command = json.loads(data["text"])

                if command.get("type") == "describe_scene":
                    # Get last frame and describe
                    if hasattr(session, "last_frame") and session.last_frame is not None:
                        description = await describe_scene(session.last_frame)
                        await websocket.send_json({
                            "type": "scene",
                            "description": description,
                        })
                    else:
                        await websocket.send_json({
                            "type": "scene",
                            "description": "No frame available",
                        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def process_frame(frame_bytes: bytes, session: SessionState) -> dict:
    """Process a single frame from the client."""
    session.frame_count += 1

    # Decode JPEG to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"type": "detection", "error": "Failed to decode frame"}

    # Store for scene description
    session.last_frame = frame

    h, w = frame.shape[:2]

    # Run detection pipeline
    person_detected = False
    face_detected = False
    user_id = None
    user_name = None
    confidence = 0.0
    is_looking = False
    attention_score = 0.0
    detections = []

    # Person detection (every frame)
    det_result = detector.detect(frame, session.frame_count)
    person_detected = det_result.has_person

    # Add all detected objects
    for det in det_result.detections:
        # Convert bbox to list of floats for JSON
        bbox = [float(x) for x in det.bbox] if det.bbox else []
        detections.append({
            "bbox": bbox,
            "confidence": float(det.confidence),
            "label": det.class_name.capitalize(),
            "class_id": det.class_id,
            "recognized": False,
            "frame_width": int(w),
            "frame_height": int(h),
        })
        # Check if any person detected
        if det.class_id == 0:
            person_detected = True

    # Face recognition (every frame when person detected for responsiveness)
    if person_detected:
        rec_result = recognizer.detect(frame, session.frame_count)

        if rec_result.has_face:
            face_detected = True
            largest = rec_result.get_largest()
            if largest:
                user_id, confidence = recognizer.identify(largest)
                if user_id:
                    user_profile = recognizer.get_user(user_id)
                    user_name = user_profile.name if user_profile else user_id

                    # Update detection with recognition info
                    for det in detections:
                        det["recognized"] = True
                        det["label"] = f"{user_name} ({confidence:.0%})"

    # Gaze tracking (every frame when person detected)
    if person_detected:
        gaze_result = attention.track(frame, session.frame_count)
        is_looking = gaze_result.is_looking_at_camera
        attention_score = gaze_result.attention_score

    # Debug logging every 30 frames
    if session.frame_count % 30 == 0:
        logger.info(f"Frame {session.frame_count}: person={person_detected}, face={face_detected}, looking={is_looking}, attention={attention_score:.2f}")

    # Update presence state
    session.presence.update(
        person_detected=person_detected,
        user_id=user_id,
        recognition_confidence=confidence,
        is_looking=is_looking,
        attention_score=attention_score,
    )

    # Get unique object labels for display
    object_labels = list(set(d["label"] for d in detections))

    # Convert numpy types to Python native types for JSON serialization
    return {
        "type": "detection",
        "frame_id": int(session.frame_count),
        "person_detected": bool(person_detected),
        "face_detected": bool(face_detected),
        "user_id": user_id,
        "user_name": user_name,
        "confidence": float(confidence),
        "is_looking": bool(is_looking),
        "attention_score": float(attention_score),
        "detections": detections,
        "objects": object_labels,  # List of detected object types
    }


async def describe_scene(frame: np.ndarray) -> str:
    """Generate scene description using VLM."""
    global scene_analyzer

    if scene_analyzer is None:
        return "Scene analysis not available"

    # Lazy load
    if not scene_analyzer.is_loaded:
        logger.info("Loading scene analyzer...")
        scene_analyzer.load()

    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        scene_analyzer.describe,
        frame,
    )

    return result.description


# Serve static files (web UI)
if os.path.exists("web"):
    app.mount("/static", StaticFiles(directory="web"), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
