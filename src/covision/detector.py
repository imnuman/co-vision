"""Person detection module using YOLO.

Provides real-time person detection with configurable models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # 0 = person in COCO
    class_name: str = "person"

    @property
    def center(self) -> tuple[int, int]:
        """Center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult:
    """Detection results for a single frame."""

    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0

    @property
    def person_count(self) -> int:
        """Number of persons detected."""
        return len(self.detections)

    @property
    def has_person(self) -> bool:
        """Whether at least one person is detected."""
        return len(self.detections) > 0

    def get_largest(self) -> Optional[Detection]:
        """Get the largest detection (closest person)."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.area)


class PersonDetector:
    """YOLO-based person detector.

    Usage:
        detector = PersonDetector(model="yolov8n")
        detector.load()

        result = detector.detect(frame)
        if result.has_person:
            print(f"Found {result.person_count} people")
    """

    # Supported models
    MODELS = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        # YOLO26 when available
        "yolo26n": "yolo11n.pt",  # Fallback until yolo26 is released in ultralytics
    }

    def __init__(
        self,
        model: str = "yolov8n",
        confidence: float = 0.5,
        device: str = "auto",
        classes: list[int] | None = None,
    ):
        """Initialize detector.

        Args:
            model: Model name (yolov8n, yolov8s, yolov10n, etc.)
            confidence: Detection confidence threshold
            device: Device (auto, cuda, mps, cpu)
            classes: COCO class IDs to detect (None = all 80 COCO classes)
        """
        self.model_name = model
        self.confidence = confidence
        self.device = device
        # None = detect all COCO classes, otherwise filter to specified classes
        self.classes = classes  # Don't default to [0], allow None for all objects

        self._model = None
        self._loaded = False

    def load(self):
        """Load the YOLO model."""
        if self._loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        model_file = self.MODELS.get(self.model_name, f"{self.model_name}.pt")
        logger.info(f"Loading YOLO model: {model_file}")

        self._model = YOLO(model_file)

        # Determine device
        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"Using device: {self.device}")
        self._loaded = True

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Detect persons in frame.

        Args:
            frame: BGR image (OpenCV format)
            frame_id: Optional frame identifier

        Returns:
            DetectionResult with all detections
        """
        if not self._loaded:
            self.load()

        import time

        start = time.time()

        results = self._model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        inference_time = (time.time() - start) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                # Get class name from model
                class_name = self._model.names.get(cls, f"class_{cls}")
                detections.append(
                    Detection(
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        confidence=conf,
                        class_id=cls,
                        class_name=class_name,
                    )
                )

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            frame_id=frame_id,
        )

    def warmup(self, size: tuple[int, int] = (640, 480)):
        """Warmup model with dummy inference.

        Args:
            size: Image size (width, height)
        """
        if not self._loaded:
            self.load()

        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("Detector warmup complete")

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self._loaded


class PersonDetectorONNX:
    """ONNX-based person detector for optimized inference.

    Use this for production deployments with TensorRT or OpenVINO.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        providers: list[str] | None = None,
    ):
        """Initialize ONNX detector.

        Args:
            model_path: Path to ONNX model
            confidence: Detection confidence threshold
            providers: ONNX Runtime execution providers
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self._session = None
        self._loaded = False

    def load(self):
        """Load ONNX model."""
        if self._loaded:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        logger.info(f"Loading ONNX model: {self.model_path}")
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )
        self._loaded = True
        logger.info(f"ONNX model loaded with providers: {self._session.get_providers()}")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """Detect persons in frame."""
        if not self._loaded:
            self.load()

        # TODO: Implement ONNX inference
        # This requires preprocessing and postprocessing specific to the model
        raise NotImplementedError("ONNX inference not yet implemented")
