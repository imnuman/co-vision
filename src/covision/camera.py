"""Camera capture module with async support.

Provides webcam capture with frame buffering for async processing.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for captured frame with metadata."""

    frame: np.ndarray
    timestamp: float
    frame_id: int


class Camera:
    """Async-capable camera capture with frame buffering.

    Usage:
        camera = Camera(device=0, width=1280, height=720)
        camera.start()

        # Sync usage
        frame = camera.read()

        # Async usage
        frame = await camera.read_async()

        camera.stop()
    """

    def __init__(
        self,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 2,
    ):
        """Initialize camera.

        Args:
            device: Camera index or video file path
            width: Capture width
            height: Capture height
            fps: Target frame rate
            buffer_size: Number of frames to buffer
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: deque[FrameData] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._last_frame: Optional[FrameData] = None

    def start(self):
        """Start camera capture in background thread."""
        if self._running:
            logger.warning("Camera already running")
            return

        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera started: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")

    def _capture_loop(self):
        """Background capture loop."""
        frame_interval = 1.0 / self.fps
        next_frame_time = time.time()

        while self._running:
            current_time = time.time()

            # Rate limiting
            if current_time < next_frame_time:
                time.sleep(0.001)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue

            self._frame_id += 1
            frame_data = FrameData(
                frame=frame,
                timestamp=current_time,
                frame_id=self._frame_id,
            )

            with self._lock:
                self._buffer.append(frame_data)
                self._last_frame = frame_data

            next_frame_time = current_time + frame_interval

    def read(self) -> Optional[FrameData]:
        """Read the latest frame (sync).

        Returns:
            FrameData or None if no frame available
        """
        with self._lock:
            return self._last_frame

    async def read_async(self) -> Optional[FrameData]:
        """Read the latest frame (async).

        Waits for a new frame if buffer is empty.
        """
        # Try to get frame immediately
        frame = self.read()
        if frame:
            return frame

        # Wait for frame with timeout
        for _ in range(100):  # 1 second timeout
            await asyncio.sleep(0.01)
            frame = self.read()
            if frame:
                return frame

        return None

    def read_buffer(self) -> list[FrameData]:
        """Read all buffered frames.

        Returns:
            List of FrameData, oldest first
        """
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
            return frames

    @property
    def is_running(self) -> bool:
        """Whether camera is currently capturing."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Total frames captured."""
        return self._frame_id

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class VideoFileCamera(Camera):
    """Camera that reads from video file instead of webcam.

    Useful for testing and development.
    """

    def __init__(
        self,
        path: str,
        loop: bool = True,
        **kwargs,
    ):
        """Initialize video file camera.

        Args:
            path: Path to video file
            loop: Whether to loop video when finished
            **kwargs: Additional Camera arguments
        """
        super().__init__(device=path, **kwargs)
        self.loop = loop
        self._path = path

    def _capture_loop(self):
        """Background capture loop with looping support."""
        frame_interval = 1.0 / self.fps

        while self._running:
            start_time = time.time()

            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.info("Video file ended")
                    break

            self._frame_id += 1
            frame_data = FrameData(
                frame=frame,
                timestamp=start_time,
                frame_id=self._frame_id,
            )

            with self._lock:
                self._buffer.append(frame_data)
                self._last_frame = frame_data

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
