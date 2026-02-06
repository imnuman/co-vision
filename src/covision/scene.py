"""Scene understanding using Vision Language Models.

Provides scene description and visual question answering.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneDescription:
    """Scene understanding result."""

    description: str = ""
    objects: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    environment: str = ""  # indoor, outdoor, office, home, etc.
    inference_time_ms: float = 0.0
    frame_id: int = 0

    @property
    def summary(self) -> str:
        """Get a one-line summary."""
        return self.description.split(".")[0] if self.description else ""


class SceneAnalyzer:
    """Scene understanding using Moondream or other small VLMs.

    Usage:
        analyzer = SceneAnalyzer()
        analyzer.load()

        result = analyzer.describe(frame)
        print(result.description)

        # Ask specific questions
        answer = analyzer.ask(frame, "What is the person doing?")
    """

    MODELS = {
        "moondream2": "vikhyatk/moondream2",
        "moondream-0.5b": "vikhyatk/moondream-0.5b",
    }

    def __init__(
        self,
        model: str = "moondream2",
        device: str = "auto",
        max_tokens: int = 100,
    ):
        """Initialize scene analyzer.

        Args:
            model: Model name (moondream2, moondream-0.5b)
            device: Device (auto, cuda, mps, cpu)
            max_tokens: Maximum tokens for generation
        """
        self.model_name = model
        self.device = device
        self.max_tokens = max_tokens

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Load the VLM model."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Run: pip install transformers torch"
            )

        model_id = self.MODELS.get(self.model_name, self.model_name)
        logger.info(f"Loading VLM model: {model_id}")

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        # Load model
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)

        self._loaded = True
        logger.info("VLM model loaded")

    def describe(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        prompt: str = "Describe what you see in this image concisely.",
    ) -> SceneDescription:
        """Generate scene description.

        Args:
            frame: BGR image (OpenCV format)
            frame_id: Optional frame identifier
            prompt: Custom prompt for description

        Returns:
            SceneDescription with analysis
        """
        if not self._loaded:
            self.load()

        import time
        from PIL import Image

        start = time.time()

        # Convert to PIL Image (RGB)
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        # Generate description
        try:
            enc_image = self._model.encode_image(pil_image)
            description = self._model.answer_question(
                enc_image,
                prompt,
                self._tokenizer,
            )
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            description = ""

        inference_time = (time.time() - start) * 1000

        # Extract objects and activities (simple parsing)
        objects = self._extract_objects(description)
        activities = self._extract_activities(description)
        environment = self._detect_environment(description)

        return SceneDescription(
            description=description,
            objects=objects,
            activities=activities,
            environment=environment,
            inference_time_ms=inference_time,
            frame_id=frame_id,
        )

    def ask(
        self,
        frame: np.ndarray,
        question: str,
    ) -> str:
        """Ask a question about the image.

        Args:
            frame: BGR image (OpenCV format)
            question: Question to ask

        Returns:
            Answer string
        """
        if not self._loaded:
            self.load()

        from PIL import Image

        # Convert to PIL Image (RGB)
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        try:
            enc_image = self._model.encode_image(pil_image)
            answer = self._model.answer_question(
                enc_image,
                question,
                self._tokenizer,
            )
            return answer
        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return ""

    def describe_for_context(
        self,
        frame: np.ndarray,
        user_present: bool = False,
        user_name: str = "",
    ) -> str:
        """Generate context-aware description for conversation.

        Optimized for providing context to an LLM conversation.

        Args:
            frame: BGR image
            user_present: Whether the known user is in frame
            user_name: Name of the user if known

        Returns:
            Context string suitable for conversation injection
        """
        if not self._loaded:
            self.load()

        prompt = "Describe the scene briefly, focusing on what the person is doing and the environment."

        if user_present and user_name:
            prompt = f"Describe what {user_name} is doing and their surroundings."

        result = self.describe(frame, prompt=prompt)

        # Format for conversation context
        if user_present:
            return f"[Visual context: {result.description}]"
        else:
            return f"[Scene: {result.description}]"

    def _extract_objects(self, description: str) -> list[str]:
        """Extract mentioned objects from description."""
        # Common objects to look for
        common_objects = [
            "computer", "laptop", "phone", "desk", "chair", "table",
            "cup", "coffee", "book", "keyboard", "mouse", "monitor",
            "window", "door", "lamp", "plant", "camera", "headphones",
            "glasses", "water", "food", "paper", "pen", "notebook",
        ]

        description_lower = description.lower()
        found = [obj for obj in common_objects if obj in description_lower]
        return found

    def _extract_activities(self, description: str) -> list[str]:
        """Extract activities from description."""
        activities = [
            "sitting", "standing", "walking", "typing", "reading",
            "writing", "eating", "drinking", "talking", "looking",
            "working", "studying", "watching", "listening", "thinking",
        ]

        description_lower = description.lower()
        found = [act for act in activities if act in description_lower]
        return found

    def _detect_environment(self, description: str) -> str:
        """Detect environment type from description."""
        description_lower = description.lower()

        if any(w in description_lower for w in ["office", "desk", "computer", "work"]):
            return "office"
        elif any(w in description_lower for w in ["kitchen", "cooking", "food"]):
            return "kitchen"
        elif any(w in description_lower for w in ["bedroom", "bed", "sleeping"]):
            return "bedroom"
        elif any(w in description_lower for w in ["living room", "couch", "sofa", "tv"]):
            return "living_room"
        elif any(w in description_lower for w in ["outdoor", "outside", "park", "street"]):
            return "outdoor"
        else:
            return "indoor"

    def warmup(self, size: tuple[int, int] = (640, 480)):
        """Warmup with dummy inference."""
        if not self._loaded:
            self.load()

        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.describe(dummy)
        logger.info("Scene analyzer warmup complete")

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self._loaded


class SceneAnalyzerOllama:
    """Scene analyzer using Ollama for local VLM inference.

    Alternative to transformers-based approach, using Ollama server.
    """

    def __init__(
        self,
        model: str = "moondream",
        host: str = "http://localhost:11434",
    ):
        """Initialize Ollama-based analyzer.

        Args:
            model: Ollama model name
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self._loaded = False

    def load(self):
        """Verify Ollama is available."""
        import requests

        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            self._loaded = True
            logger.info(f"Ollama connection verified: {self.host}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")

    def describe(self, frame: np.ndarray, frame_id: int = 0) -> SceneDescription:
        """Generate scene description via Ollama."""
        if not self._loaded:
            self.load()

        import base64
        import time
        import requests
        import cv2

        start = time.time()

        # Encode image as base64
        _, buffer = cv2.imencode(".jpg", frame)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        # Call Ollama API
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Describe this image concisely.",
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            description = response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            description = ""

        inference_time = (time.time() - start) * 1000

        return SceneDescription(
            description=description,
            inference_time_ms=inference_time,
            frame_id=frame_id,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded
