"""Main vision pipeline orchestrating all components.

This is the primary interface for CoVision.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml

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
        self._loop: asyncio.AbstractEventLoop | None = None

        # State
        self._user_present = False
        self._user_looking = False
        self._current_user_id: str | None = None
        self._last_scene: str = ""

        # Components (lazy loaded)
        self._camera = None
        self._detector = None
        self._recognizer = None
        self._attention = None
        self._scene = None

        logger.info("VisionSystem initialized")

    def _load_config(self, config_path: str | Path | None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Use defaults
            return {
                "camera": {"device": 0, "width": 1280, "height": 720, "fps": 30},
                "detection": {"model": "yolo26n", "confidence": 0.5, "device": "auto"},
                "recognition": {"model": "buffalo_l", "threshold": 0.5},
                "gaze": {"attention_threshold": 0.3, "smoothing_window": 5},
                "scene": {"model": "moondream2", "trigger": "on_demand"},
            }

        with open(config_path) as f:
            return yaml.safe_load(f)

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
        return self._user_present

    @property
    def user_looking(self) -> bool:
        """Whether the user is looking at the camera."""
        return self._user_looking

    @property
    def current_user(self) -> str | None:
        """ID of the currently recognized user."""
        return self._current_user_id

    def describe_scene(self) -> str:
        """Get current scene description.

        Triggers VLM analysis if not recently updated.
        """
        # TODO: Implement with Moondream
        return self._last_scene or "Scene analysis not available"

    def get_context(self) -> dict:
        """Get current vision context for LLM integration.

        Returns dict suitable for injecting into conversation context.
        """
        return {
            "user_present": self._user_present,
            "user_looking": self._user_looking,
            "user_id": self._current_user_id,
            "scene": self._last_scene,
        }

    def start(self):
        """Start the vision system (blocking)."""
        self._running = True
        logger.info("Starting VisionSystem")

        try:
            asyncio.run(self._run_loop())
        except KeyboardInterrupt:
            logger.info("VisionSystem stopped by user")
        finally:
            self._running = False

    async def start_async(self):
        """Start the vision system (non-blocking, for async contexts)."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        asyncio.create_task(self._run_loop())

    def stop(self):
        """Stop the vision system."""
        self._running = False
        logger.info("Stopping VisionSystem")

    async def _run_loop(self):
        """Main processing loop."""
        logger.info("Vision loop started")

        # TODO: Initialize components
        # self._camera = Camera(self.config["camera"])
        # self._detector = Detector(self.config["detection"])
        # etc.

        frame_count = 0

        while self._running:
            try:
                # TODO: Capture frame
                # frame = await self._camera.read_async()

                # Placeholder for development
                await asyncio.sleep(1 / 30)  # 30 FPS
                frame_count += 1

                # TODO: Run detection pipeline
                # persons = await self._detect_persons(frame)
                # if persons:
                #     faces = await self._detect_faces(frame)
                #     ...

                # Emit periodic status (for development)
                if frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    logger.debug(f"Processed {frame_count} frames")

            except Exception as e:
                logger.error(f"Error in vision loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Vision loop stopped")

    async def _detect_persons(self, frame):
        """Run person detection on frame."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._detector.detect,
            frame
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self._executor.shutdown(wait=True)
