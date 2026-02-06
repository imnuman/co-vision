"""Presence state machine with hysteresis.

Manages user presence detection with temporal smoothing to avoid flickering.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from covision.events import (
    EventEmitter,
    UserArrivedEvent,
    UserLeftEvent,
    UserLookingEvent,
)

logger = logging.getLogger(__name__)


class PresenceState(Enum):
    """User presence states."""

    ABSENT = "absent"  # No one in frame
    DETECTED = "detected"  # Person detected, not yet recognized
    PRESENT = "present"  # Known user recognized
    LOOKING = "looking"  # User is looking at camera


@dataclass
class PresenceInfo:
    """Current presence information."""

    state: PresenceState = PresenceState.ABSENT
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    confidence: float = 0.0
    is_looking: bool = False
    attention_score: float = 0.0
    duration_seconds: float = 0.0
    last_seen: float = 0.0

    @property
    def is_present(self) -> bool:
        """Whether a known user is present."""
        return self.state in (PresenceState.PRESENT, PresenceState.LOOKING)

    @property
    def is_paying_attention(self) -> bool:
        """Whether the user is actively looking at camera."""
        return self.state == PresenceState.LOOKING


class PresenceManager:
    """Manages user presence detection with hysteresis.

    Prevents rapid state changes by requiring consistent detection
    over multiple frames before triggering events.

    Usage:
        presence = PresenceManager(events=event_emitter)

        # In your processing loop
        presence.update(
            person_detected=True,
            user_id="user1",
            recognition_confidence=0.85,
            is_looking=True,
            attention_score=0.9,
        )

        # Check current state
        if presence.info.is_present:
            print(f"User {presence.info.user_name} is here")
    """

    def __init__(
        self,
        events: Optional[EventEmitter] = None,
        arrival_frames: int = 3,
        departure_frames: int = 30,
        attention_frames: int = 5,
        recognition_threshold: float = 0.5,
    ):
        """Initialize presence manager.

        Args:
            events: Event emitter for presence events
            arrival_frames: Frames user must be present before "arrived" event
            departure_frames: Frames user must be absent before "left" event
            attention_frames: Frames user must be looking before "looking" event
            recognition_threshold: Minimum confidence for recognition
        """
        self.events = events or EventEmitter()
        self.arrival_frames = arrival_frames
        self.departure_frames = departure_frames
        self.attention_frames = attention_frames
        self.recognition_threshold = recognition_threshold

        # State
        self._state = PresenceState.ABSENT
        self._user_id: Optional[str] = None
        self._user_name: Optional[str] = None
        self._confidence: float = 0.0
        self._is_looking: bool = False
        self._attention_score: float = 0.0

        # Counters for hysteresis
        self._present_count: int = 0
        self._absent_count: int = 0
        self._looking_count: int = 0
        self._not_looking_count: int = 0

        # Timing
        self._arrival_time: float = 0.0
        self._last_update: float = 0.0

        # User name lookup
        self._user_names: dict[str, str] = {}

    def register_user(self, user_id: str, name: str):
        """Register a user name for display."""
        self._user_names[user_id] = name

    def update(
        self,
        person_detected: bool,
        user_id: Optional[str] = None,
        recognition_confidence: float = 0.0,
        is_looking: bool = False,
        attention_score: float = 0.0,
    ):
        """Update presence state with new detection results.

        Args:
            person_detected: Whether any person is detected
            user_id: Recognized user ID (None if unknown)
            recognition_confidence: Face recognition confidence
            is_looking: Whether user is looking at camera
            attention_score: Attention/gaze score (0-1)
        """
        now = time.time()
        self._last_update = now

        # Determine if this is a valid recognition
        is_recognized = (
            user_id is not None
            and recognition_confidence >= self.recognition_threshold
        )

        # Update counters based on detection
        if person_detected and is_recognized:
            self._present_count += 1
            self._absent_count = 0
            self._user_id = user_id
            self._confidence = recognition_confidence
        elif person_detected:
            # Person detected but not recognized
            self._present_count = 0
            self._absent_count = 0
        else:
            self._present_count = 0
            self._absent_count += 1

        # Update attention counters
        if is_looking and person_detected:
            self._looking_count += 1
            self._not_looking_count = 0
        else:
            self._looking_count = 0
            self._not_looking_count += 1

        self._is_looking = is_looking
        self._attention_score = attention_score

        # State transitions
        old_state = self._state
        new_state = self._compute_state()

        if new_state != old_state:
            self._handle_state_transition(old_state, new_state, now)

        self._state = new_state

    def _compute_state(self) -> PresenceState:
        """Compute new state based on counters."""
        # Check for departure first
        if self._absent_count >= self.departure_frames:
            return PresenceState.ABSENT

        # Check for arrival
        if self._present_count >= self.arrival_frames:
            # Check for attention
            if self._looking_count >= self.attention_frames:
                return PresenceState.LOOKING
            return PresenceState.PRESENT

        # Maintain current state during hysteresis window
        if self._state in (PresenceState.PRESENT, PresenceState.LOOKING):
            if self._looking_count >= self.attention_frames:
                return PresenceState.LOOKING
            return PresenceState.PRESENT

        return PresenceState.ABSENT

    def _handle_state_transition(
        self,
        old_state: PresenceState,
        new_state: PresenceState,
        timestamp: float,
    ):
        """Handle state transition and emit events."""
        logger.info(f"Presence state: {old_state.value} -> {new_state.value}")

        # User arrived
        if old_state == PresenceState.ABSENT and new_state in (
            PresenceState.PRESENT,
            PresenceState.LOOKING,
        ):
            self._arrival_time = timestamp
            self._user_name = self._user_names.get(self._user_id, self._user_id)

            self.events.emit(
                UserArrivedEvent(
                    user_id=self._user_id or "",
                    confidence=self._confidence,
                )
            )
            logger.info(f"User arrived: {self._user_name}")

        # User left
        elif new_state == PresenceState.ABSENT and old_state in (
            PresenceState.PRESENT,
            PresenceState.LOOKING,
        ):
            duration = timestamp - self._arrival_time if self._arrival_time else 0

            self.events.emit(
                UserLeftEvent(
                    user_id=self._user_id or "",
                    duration_seconds=duration,
                )
            )
            logger.info(f"User left after {duration:.1f}s")

            # Reset user info
            self._user_id = None
            self._user_name = None
            self._arrival_time = 0.0

        # User started looking
        elif new_state == PresenceState.LOOKING and old_state == PresenceState.PRESENT:
            self.events.emit(
                UserLookingEvent(
                    user_id=self._user_id or "",
                    gaze_direction=(0.0, 0.0),
                )
            )
            logger.debug("User is now looking at camera")

    @property
    def info(self) -> PresenceInfo:
        """Get current presence information."""
        duration = 0.0
        if self._arrival_time and self._state != PresenceState.ABSENT:
            duration = time.time() - self._arrival_time

        return PresenceInfo(
            state=self._state,
            user_id=self._user_id,
            user_name=self._user_name,
            confidence=self._confidence,
            is_looking=self._is_looking,
            attention_score=self._attention_score,
            duration_seconds=duration,
            last_seen=self._last_update,
        )

    @property
    def state(self) -> PresenceState:
        """Current presence state."""
        return self._state

    @property
    def is_present(self) -> bool:
        """Whether a known user is present."""
        return self._state in (PresenceState.PRESENT, PresenceState.LOOKING)

    @property
    def is_looking(self) -> bool:
        """Whether the user is looking at camera."""
        return self._state == PresenceState.LOOKING

    def reset(self):
        """Reset all state."""
        self._state = PresenceState.ABSENT
        self._user_id = None
        self._user_name = None
        self._confidence = 0.0
        self._is_looking = False
        self._attention_score = 0.0
        self._present_count = 0
        self._absent_count = 0
        self._looking_count = 0
        self._not_looking_count = 0
        self._arrival_time = 0.0
        logger.info("Presence state reset")
