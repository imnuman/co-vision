"""Event system for CoVision.

Provides a simple pub/sub event emitter for presence events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from collections import defaultdict


@dataclass
class Event:
    """Base event class."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserArrivedEvent(Event):
    """Emitted when a known user enters the frame."""

    name: str = "user_arrived"
    user_id: str = ""
    confidence: float = 0.0


@dataclass
class UserLeftEvent(Event):
    """Emitted when the user leaves the frame."""

    name: str = "user_left"
    user_id: str = ""
    duration_seconds: float = 0.0


@dataclass
class UserLookingEvent(Event):
    """Emitted when the user is looking at the camera."""

    name: str = "user_looking"
    user_id: str = ""
    gaze_direction: tuple[float, float] = (0.0, 0.0)


@dataclass
class SceneUpdateEvent(Event):
    """Emitted when scene description is updated."""

    name: str = "scene_update"
    description: str = ""
    objects: list[str] = field(default_factory=list)


class EventEmitter:
    """Simple event emitter for pub/sub pattern."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event_name: str, handler: Callable | None = None):
        """Register an event handler.

        Can be used as a decorator:
            @emitter.on("user_arrived")
            def handle_arrival(event):
                ...

        Or directly:
            emitter.on("user_arrived", handle_arrival)
        """
        def decorator(fn: Callable) -> Callable:
            self._handlers[event_name].append(fn)
            return fn

        if handler is not None:
            self._handlers[event_name].append(handler)
            return handler

        return decorator

    def off(self, event_name: str, handler: Callable):
        """Unregister an event handler."""
        if event_name in self._handlers:
            self._handlers[event_name] = [
                h for h in self._handlers[event_name] if h != handler
            ]

    def emit(self, event: Event):
        """Emit an event to all registered handlers."""
        for handler in self._handlers.get(event.name, []):
            try:
                handler(event)
            except Exception as e:
                # Log but don't crash on handler errors
                print(f"Error in event handler for {event.name}: {e}")

    async def emit_async(self, event: Event):
        """Emit an event, awaiting async handlers."""
        import asyncio

        for handler in self._handlers.get(event.name, []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Error in event handler for {event.name}: {e}")
