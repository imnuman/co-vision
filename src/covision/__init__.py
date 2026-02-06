"""CoVision - Computer Vision for AI Companions.

Real-time presence detection, face recognition, gaze tracking, and scene understanding.
"""

__version__ = "0.1.0"

from covision.events import Event, EventEmitter, UserArrivedEvent, UserLeftEvent, UserLookingEvent
from covision.pipeline import VisionSystem
from covision.camera import Camera
from covision.detector import PersonDetector, Detection, DetectionResult
from covision.recognizer import FaceRecognizer, Face, RecognitionResult
from covision.attention import AttentionTracker, GazeResult, GazeDirection
from covision.scene import SceneAnalyzer, SceneDescription
from covision.presence import PresenceManager, PresenceState, PresenceInfo

__all__ = [
    # Main interface
    "VisionSystem",
    # Components
    "Camera",
    "PersonDetector",
    "FaceRecognizer",
    "AttentionTracker",
    "SceneAnalyzer",
    "PresenceManager",
    # Data classes
    "Detection",
    "DetectionResult",
    "Face",
    "RecognitionResult",
    "GazeResult",
    "GazeDirection",
    "SceneDescription",
    "PresenceState",
    "PresenceInfo",
    # Events
    "Event",
    "EventEmitter",
    "UserArrivedEvent",
    "UserLeftEvent",
    "UserLookingEvent",
    # Version
    "__version__",
]
