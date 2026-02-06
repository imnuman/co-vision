"""CoVision - Computer Vision for AI Companions.

Real-time presence detection, face recognition, gaze tracking, and scene understanding.
"""

__version__ = "0.1.0"

from covision.events import Event, EventEmitter
from covision.pipeline import VisionSystem

__all__ = [
    "VisionSystem",
    "Event",
    "EventEmitter",
    "__version__",
]
