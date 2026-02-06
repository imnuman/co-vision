"""Gaze and attention tracking using MediaPipe.

Provides real-time gaze direction and attention detection.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GazeDirection(Enum):
    """Discrete gaze directions."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    AWAY = "away"  # Not looking at camera


@dataclass
class GazeResult:
    """Gaze tracking result for a single frame."""

    # Continuous gaze values (-1 to 1, 0 = center)
    horizontal: float = 0.0  # Negative = left, positive = right
    vertical: float = 0.0  # Negative = up, positive = down

    # Discrete direction
    direction: GazeDirection = GazeDirection.CENTER

    # Attention metrics
    is_looking_at_camera: bool = False
    attention_score: float = 0.0  # 0-1, higher = more focused on camera

    # Eye state
    left_eye_open: bool = True
    right_eye_open: bool = True
    is_blinking: bool = False

    # Head pose (if available)
    head_yaw: float = 0.0  # Left/right rotation
    head_pitch: float = 0.0  # Up/down rotation
    head_roll: float = 0.0  # Tilt

    # Metadata
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    frame_id: int = 0


class AttentionTracker:
    """Gaze and attention tracking using MediaPipe Face Mesh.

    Usage:
        tracker = AttentionTracker()
        tracker.load()

        result = tracker.track(frame)
        if result.is_looking_at_camera:
            print("User is paying attention!")
    """

    # MediaPipe landmark indices
    # Left eye
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_IRIS_CENTER = 468  # With refine_landmarks=True

    # Right eye
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_IRIS_CENTER = 473  # With refine_landmarks=True

    # Nose tip for head pose reference
    NOSE_TIP = 1

    def __init__(
        self,
        attention_threshold: float = 0.3,
        smoothing_window: int = 5,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize attention tracker.

        Args:
            attention_threshold: How centered gaze must be to count as "looking"
            smoothing_window: Number of frames for temporal smoothing
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
        """
        self.attention_threshold = attention_threshold
        self.smoothing_window = smoothing_window
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._face_mesh = None
        self._loaded = False

        # Smoothing buffers
        self._horizontal_buffer: list[float] = []
        self._vertical_buffer: list[float] = []

    def load(self):
        """Load MediaPipe Face Mesh."""
        if self._loaded:
            return

        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe not installed. Run: pip install mediapipe"
            )

        logger.info("Loading MediaPipe Face Mesh")

        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self._loaded = True
        logger.info("MediaPipe Face Mesh loaded")

    def track(self, frame: np.ndarray, frame_id: int = 0) -> GazeResult:
        """Track gaze and attention in frame.

        Args:
            frame: BGR image (OpenCV format)
            frame_id: Optional frame identifier

        Returns:
            GazeResult with gaze direction and attention metrics
        """
        if not self._loaded:
            self.load()

        import time
        import cv2

        start = time.time()

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        inference_time = (time.time() - start) * 1000

        if not results.multi_face_landmarks:
            return GazeResult(
                direction=GazeDirection.AWAY,
                is_looking_at_camera=False,
                attention_score=0.0,
                inference_time_ms=inference_time,
                frame_id=frame_id,
            )

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Extract gaze direction from iris position
        horizontal, vertical = self._compute_gaze_direction(landmarks, w, h)

        # Apply temporal smoothing
        horizontal = self._smooth_value(horizontal, self._horizontal_buffer)
        vertical = self._smooth_value(vertical, self._vertical_buffer)

        # Determine discrete direction
        direction = self._classify_direction(horizontal, vertical)

        # Compute attention score (inverse of distance from center)
        gaze_magnitude = np.sqrt(horizontal**2 + vertical**2)
        attention_score = max(0.0, 1.0 - gaze_magnitude)

        # Determine if looking at camera
        is_looking = gaze_magnitude < self.attention_threshold

        # Eye aspect ratio for blink detection
        left_ear = self._eye_aspect_ratio(landmarks, "left", w, h)
        right_ear = self._eye_aspect_ratio(landmarks, "right", w, h)
        is_blinking = left_ear < 0.2 or right_ear < 0.2

        # Head pose estimation
        head_yaw, head_pitch, head_roll = self._estimate_head_pose(landmarks, w, h)

        return GazeResult(
            horizontal=horizontal,
            vertical=vertical,
            direction=direction,
            is_looking_at_camera=is_looking,
            attention_score=attention_score,
            left_eye_open=left_ear > 0.2,
            right_eye_open=right_ear > 0.2,
            is_blinking=is_blinking,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            confidence=1.0,
            inference_time_ms=inference_time,
            frame_id=frame_id,
        )

    def _compute_gaze_direction(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Compute gaze direction from iris position.

        Returns:
            Tuple of (horizontal, vertical) in range [-1, 1]
        """
        # Get iris centers (available with refine_landmarks=True)
        left_iris = landmarks.landmark[self.LEFT_IRIS_CENTER]
        right_iris = landmarks.landmark[self.RIGHT_IRIS_CENTER]

        # Get eye corners for reference
        left_inner = landmarks.landmark[self.LEFT_EYE_INNER]
        left_outer = landmarks.landmark[self.LEFT_EYE_OUTER]
        right_inner = landmarks.landmark[self.RIGHT_EYE_INNER]
        right_outer = landmarks.landmark[self.RIGHT_EYE_OUTER]

        # Compute iris position relative to eye corners
        # Left eye
        left_eye_width = abs(left_outer.x - left_inner.x)
        left_iris_pos = (left_iris.x - left_outer.x) / left_eye_width if left_eye_width > 0 else 0.5

        # Right eye
        right_eye_width = abs(right_outer.x - right_inner.x)
        right_iris_pos = (right_iris.x - right_inner.x) / right_eye_width if right_eye_width > 0 else 0.5

        # Average both eyes for horizontal gaze
        # Map from [0, 1] to [-1, 1] where 0.5 is center
        horizontal = ((left_iris_pos + right_iris_pos) / 2 - 0.5) * 2

        # Vertical gaze from iris position relative to eye height
        left_top = landmarks.landmark[self.LEFT_EYE_TOP]
        left_bottom = landmarks.landmark[self.LEFT_EYE_BOTTOM]
        left_eye_height = abs(left_bottom.y - left_top.y)
        left_vert_pos = (left_iris.y - left_top.y) / left_eye_height if left_eye_height > 0 else 0.5

        right_top = landmarks.landmark[self.RIGHT_EYE_TOP]
        right_bottom = landmarks.landmark[self.RIGHT_EYE_BOTTOM]
        right_eye_height = abs(right_bottom.y - right_top.y)
        right_vert_pos = (right_iris.y - right_top.y) / right_eye_height if right_eye_height > 0 else 0.5

        vertical = ((left_vert_pos + right_vert_pos) / 2 - 0.5) * 2

        return horizontal, vertical

    def _eye_aspect_ratio(
        self,
        landmarks,
        eye: str,
        width: int,
        height: int,
    ) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        if eye == "left":
            top = landmarks.landmark[self.LEFT_EYE_TOP]
            bottom = landmarks.landmark[self.LEFT_EYE_BOTTOM]
            inner = landmarks.landmark[self.LEFT_EYE_INNER]
            outer = landmarks.landmark[self.LEFT_EYE_OUTER]
        else:
            top = landmarks.landmark[self.RIGHT_EYE_TOP]
            bottom = landmarks.landmark[self.RIGHT_EYE_BOTTOM]
            inner = landmarks.landmark[self.RIGHT_EYE_INNER]
            outer = landmarks.landmark[self.RIGHT_EYE_OUTER]

        # Vertical distance
        vert_dist = abs(bottom.y - top.y) * height

        # Horizontal distance
        horiz_dist = abs(outer.x - inner.x) * width

        if horiz_dist == 0:
            return 1.0

        return vert_dist / horiz_dist

    def _estimate_head_pose(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        """Estimate head pose from facial landmarks.

        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        # Simplified head pose from key points
        nose = landmarks.landmark[self.NOSE_TIP]
        left_eye = landmarks.landmark[self.LEFT_EYE_OUTER]
        right_eye = landmarks.landmark[self.RIGHT_EYE_OUTER]

        # Yaw from nose position relative to eye center
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (nose.x - eye_center_x) * 90  # Approximate degrees

        # Pitch from nose position relative to eye level
        eye_center_y = (left_eye.y + right_eye.y) / 2
        pitch = (nose.y - eye_center_y) * 90

        # Roll from eye tilt
        eye_diff_y = right_eye.y - left_eye.y
        eye_diff_x = right_eye.x - left_eye.x
        roll = np.degrees(np.arctan2(eye_diff_y, eye_diff_x))

        return float(yaw), float(pitch), float(roll)

    def _classify_direction(
        self,
        horizontal: float,
        vertical: float,
    ) -> GazeDirection:
        """Classify gaze into discrete direction."""
        threshold = self.attention_threshold

        if abs(horizontal) < threshold and abs(vertical) < threshold:
            return GazeDirection.CENTER
        elif horizontal < -threshold:
            return GazeDirection.LEFT
        elif horizontal > threshold:
            return GazeDirection.RIGHT
        elif vertical < -threshold:
            return GazeDirection.UP
        elif vertical > threshold:
            return GazeDirection.DOWN
        else:
            return GazeDirection.CENTER

    def _smooth_value(self, value: float, buffer: list[float]) -> float:
        """Apply temporal smoothing to a value."""
        buffer.append(value)
        if len(buffer) > self.smoothing_window:
            buffer.pop(0)
        return sum(buffer) / len(buffer)

    def reset_smoothing(self):
        """Reset smoothing buffers."""
        self._horizontal_buffer.clear()
        self._vertical_buffer.clear()

    def warmup(self, size: tuple[int, int] = (640, 480)):
        """Warmup with dummy inference."""
        if not self._loaded:
            self.load()

        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.track(dummy)
        logger.info("Attention tracker warmup complete")

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self._loaded
