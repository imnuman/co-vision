"""Main vision pipeline orchestrating all components.

This is the primary interface for CoVision.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from covision.camera import Camera
from covision.detector import PersonDetector, DetectionResult
from covision.recognizer import FaceRecognizer, RecognitionResult
from covision.attention import AttentionTracker, GazeResult
from covision.scene import SceneAnalyzer, SceneDescription
from covision.presence import PresenceManager, PresenceState
from covision.events import (
    EventEmitter,
    UserArrivedEvent,
    UserLeftEvent,
    UserLookingEvent,
    SceneUpdateEvent,
)

logger = logging.getLogger(__name__)


class VisionSystem:
    """Main vision system coordinating all components.

    Orchestrates camera capture, person detection, face recognition,
    gaze tracking, and scene understanding into a unified pipeline.

    Usage:
        vision = VisionSystem()

        @vision.on("user_arrived")
        def on_arrival(event):
            print(f"Welcome, {event.user_id}!")

        vision.start()
    """

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the vision system.

        Args:
            config_path: Path to YAML config file. Uses defaults if None.
        """
        self.config = self._load_config(config_path)
        self.events = EventEmitter()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Frame counters for tiered processing
        self._frame_count = 0
        self._last_scene_time = 0.0

        # Scene cache
        self._last_scene: str = ""
        self._last_scene_result: Optional[SceneDescription] = None

        # Components (lazy loaded)
        self.camera: Optional[Camera] = None
        self.detector: Optional[PersonDetector] = None
        self.recognizer: Optional[FaceRecognizer] = None
        self.attention: Optional[AttentionTracker] = None
        self.scene_analyzer: Optional[SceneAnalyzer] = None
        self.presence: Optional[PresenceManager] = None

        # Processing intervals from config
        pipeline_config = self.config.get("pipeline", {})
        self._detection_interval = pipeline_config.get("detection_interval", 1)
        self._face_interval = pipeline_config.get("face_interval", 3)
        self._recognition_interval = pipeline_config.get("recognition_interval", 6)
        self._gaze_interval = pipeline_config.get("gaze_interval", 2)

        logger.info("VisionSystem initialized")

    def _load_config(self, config_path: str | Path | None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            return {
                "camera": {"device": 0, "width": 1280, "height": 720, "fps": 30},
                "detection": {"model": "yolov8n", "confidence": 0.5, "device": "auto"},
                "recognition": {
                    "model": "buffalo_l",
                    "threshold": 0.5,
                    "embeddings_path": "models/embeddings",
                },
                "gaze": {
                    "attention_threshold": 0.3,
                    "smoothing_window": 5,
                    "enabled": True,
                },
                "scene": {
                    "model": "moondream2",
                    "trigger": "on_demand",
                    "interval": 5.0,
                    "enabled": True,
                },
                "presence": {
                    "arrival_frames": 3,
                    "departure_frames": 30,
                },
                "pipeline": {
                    "detection_interval": 1,
                    "face_interval": 3,
                    "recognition_interval": 6,
                    "gaze_interval": 2,
                },
            }

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _init_components(self):
        """Initialize all vision components."""
        logger.info("Initializing vision components...")

        # Camera
        cam_config = self.config.get("camera", {})
        self.camera = Camera(
            device=cam_config.get("device", 0),
            width=cam_config.get("width", 1280),
            height=cam_config.get("height", 720),
            fps=cam_config.get("fps", 30),
        )

        # Person detector
        det_config = self.config.get("detection", {})
        self.detector = PersonDetector(
            model=det_config.get("model", "yolov8n"),
            confidence=det_config.get("confidence", 0.5),
            device=det_config.get("device", "auto"),
        )

        # Face recognizer
        rec_config = self.config.get("recognition", {})
        self.recognizer = FaceRecognizer(
            model=rec_config.get("model", "buffalo_l"),
            threshold=rec_config.get("threshold", 0.5),
            embeddings_path=rec_config.get("embeddings_path"),
            device=det_config.get("device", "auto"),
        )

        # Attention tracker
        gaze_config = self.config.get("gaze", {})
        if gaze_config.get("enabled", True):
            self.attention = AttentionTracker(
                attention_threshold=gaze_config.get("attention_threshold", 0.3),
                smoothing_window=gaze_config.get("smoothing_window", 5),
            )

        # Scene analyzer
        scene_config = self.config.get("scene", {})
        if scene_config.get("enabled", True):
            self.scene_analyzer = SceneAnalyzer(
                model=scene_config.get("model", "moondream2"),
                device=det_config.get("device", "auto"),
            )

        # Presence manager
        presence_config = self.config.get("presence", {})
        self.presence = PresenceManager(
            events=self.events,
            arrival_frames=presence_config.get("arrival_frames", 3),
            departure_frames=presence_config.get("departure_frames", 30),
        )

        logger.info("Vision components initialized")

    def on(self, event_name: str, handler: Callable | None = None):
        """Register an event handler.

        Can be used as decorator or direct call.

        Events:
            - user_arrived: User entered frame and recognized
            - user_left: User left the frame
            - user_looking: User is looking at camera
            - scene_update: Scene description updated
        """
        return self.events.on(event_name, handler)

    @property
    def user_present(self) -> bool:
        """Whether a known user is currently in frame."""
        if self.presence:
            return self.presence.is_present
        return False

    @property
    def user_looking(self) -> bool:
        """Whether the user is looking at the camera."""
        if self.presence:
            return self.presence.is_looking
        return False

    @property
    def current_user(self) -> str | None:
        """ID of the currently recognized user."""
        if self.presence:
            return self.presence.info.user_id
        return None

    def describe_scene(self, force: bool = False) -> str:
        """Get current scene description.

        Args:
            force: Force new analysis even if cached

        Returns:
            Scene description string
        """
        if not force and self._last_scene:
            return self._last_scene

        if self.scene_analyzer and self.camera:
            frame_data = self.camera.read()
            if frame_data:
                result = self.scene_analyzer.describe(frame_data.frame)
                self._last_scene = result.description
                self._last_scene_result = result
                return result.description

        return "Scene analysis not available"

    def get_context(self) -> dict:
        """Get current vision context for LLM integration.

        Returns dict suitable for injecting into conversation context.
        """
        info = self.presence.info if self.presence else None

        return {
            "user_present": info.is_present if info else False,
            "user_looking": info.is_paying_attention if info else False,
            "user_id": info.user_id if info else None,
            "user_name": info.user_name if info else None,
            "attention_score": info.attention_score if info else 0.0,
            "duration_seconds": info.duration_seconds if info else 0.0,
            "scene": self._last_scene,
        }

    def enroll_user(self, user_id: str, name: str, num_frames: int = 10) -> bool:
        """Enroll a new user by capturing their face.

        Args:
            user_id: Unique user identifier
            name: Display name
            num_frames: Number of frames to capture

        Returns:
            True if enrollment successful
        """
        if not self.recognizer:
            logger.error("Recognizer not initialized")
            return False

        if not self.camera:
            logger.error("Camera not initialized")
            return False

        # Ensure components are loaded
        self.recognizer.load()

        frames = []
        logger.info(f"Capturing {num_frames} frames for enrollment...")

        for i in range(num_frames * 3):  # Capture more frames, select best
            frame_data = self.camera.read()
            if frame_data:
                frames.append(frame_data.frame)
            time.sleep(0.1)

            if len(frames) >= num_frames:
                break

        if len(frames) < 3:
            logger.error("Not enough frames captured")
            return False

        success = self.recognizer.enroll(user_id, name, frames)

        if success and self.presence:
            self.presence.register_user(user_id, name)

        return success

    def start(self):
        """Start the vision system (blocking)."""
        self._running = True
        logger.info("Starting VisionSystem")

        try:
            asyncio.run(self._run_loop())
        except KeyboardInterrupt:
            logger.info("VisionSystem stopped by user")
        finally:
            self._cleanup()

    async def start_async(self):
        """Start the vision system (non-blocking, for async contexts)."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        asyncio.create_task(self._run_loop())

    def stop(self):
        """Stop the vision system."""
        self._running = False
        logger.info("Stopping VisionSystem")

    def _cleanup(self):
        """Clean up resources."""
        self._running = False

        if self.camera:
            self.camera.stop()

        self._executor.shutdown(wait=False)
        logger.info("VisionSystem cleanup complete")

    async def _run_loop(self):
        """Main processing loop."""
        # Initialize components
        self._init_components()

        # Start camera
        self.camera.start()

        # Warmup models
        logger.info("Warming up models...")
        await self._warmup_models()

        logger.info("Vision loop started")

        while self._running:
            try:
                # Get frame
                frame_data = self.camera.read()
                if not frame_data:
                    await asyncio.sleep(0.01)
                    continue

                frame = frame_data.frame
                self._frame_count += 1

                # Run tiered detection pipeline
                await self._process_frame(frame, frame_data.frame_id)

                # Yield to event loop
                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in vision loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Vision loop stopped")

    async def _warmup_models(self):
        """Warmup models with dummy inference."""
        loop = asyncio.get_event_loop()

        # Warmup in parallel
        tasks = []

        if self.detector:
            tasks.append(
                loop.run_in_executor(self._executor, self.detector.warmup)
            )

        if self.recognizer:
            tasks.append(
                loop.run_in_executor(self._executor, self.recognizer.warmup)
            )

        if self.attention:
            tasks.append(
                loop.run_in_executor(self._executor, self.attention.warmup)
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Model warmup complete")

    async def _process_frame(self, frame: np.ndarray, frame_id: int):
        """Process a single frame through the pipeline."""
        loop = asyncio.get_event_loop()

        person_detected = False
        user_id = None
        recognition_confidence = 0.0
        is_looking = False
        attention_score = 0.0

        # Stage 1: Person detection (every frame or detection_interval)
        if self._frame_count % self._detection_interval == 0:
            detection_result = await loop.run_in_executor(
                self._executor,
                self.detector.detect,
                frame,
                frame_id,
            )
            person_detected = detection_result.has_person

        # Stage 2: Face recognition (less frequent)
        if person_detected and self._frame_count % self._face_interval == 0:
            recognition_result = await loop.run_in_executor(
                self._executor,
                self.recognizer.detect,
                frame,
                frame_id,
            )

            if recognition_result.has_face:
                largest_face = recognition_result.get_largest()
                if largest_face:
                    user_id, recognition_confidence = self.recognizer.identify(
                        largest_face
                    )

        # Stage 3: Gaze tracking
        if person_detected and self.attention and self._frame_count % self._gaze_interval == 0:
            gaze_result = await loop.run_in_executor(
                self._executor,
                self.attention.track,
                frame,
                frame_id,
            )
            is_looking = gaze_result.is_looking_at_camera
            attention_score = gaze_result.attention_score

        # Update presence state
        if self.presence:
            self.presence.update(
                person_detected=person_detected,
                user_id=user_id,
                recognition_confidence=recognition_confidence,
                is_looking=is_looking,
                attention_score=attention_score,
            )

        # Stage 4: Scene analysis (on-demand or periodic)
        scene_config = self.config.get("scene", {})
        if (
            self.scene_analyzer
            and scene_config.get("trigger") == "periodic"
            and person_detected
        ):
            interval = scene_config.get("interval", 5.0)
            now = time.time()
            if now - self._last_scene_time >= interval:
                self._last_scene_time = now
                scene_result = await loop.run_in_executor(
                    self._executor,
                    self.scene_analyzer.describe,
                    frame,
                    frame_id,
                )
                self._last_scene = scene_result.description
                self._last_scene_result = scene_result

                # Emit scene update event
                self.events.emit(
                    SceneUpdateEvent(
                        description=scene_result.description,
                        objects=scene_result.objects,
                    )
                )

        # Log periodic status
        if self._frame_count % 300 == 0:
            info = self.presence.info if self.presence else None
            logger.debug(
                f"Frame {self._frame_count}: "
                f"present={info.is_present if info else False}, "
                f"looking={info.is_paying_attention if info else False}"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
