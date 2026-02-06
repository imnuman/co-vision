"""Face recognition module using InsightFace.

Provides face detection and recognition with ArcFace embeddings.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Detected face with embedding."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None  # 5-point facial landmarks
    age: Optional[int] = None
    gender: Optional[str] = None

    @property
    def center(self) -> tuple[int, int]:
        """Center point of face bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def size(self) -> tuple[int, int]:
        """Size of face (width, height)."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)


@dataclass
class RecognitionResult:
    """Recognition result for a single frame."""

    faces: list[Face] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0

    @property
    def face_count(self) -> int:
        """Number of faces detected."""
        return len(self.faces)

    @property
    def has_face(self) -> bool:
        """Whether at least one face is detected."""
        return len(self.faces) > 0

    def get_largest(self) -> Optional[Face]:
        """Get the largest face (closest person)."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.size[0] * f.size[1])


@dataclass
class UserProfile:
    """Stored user profile with face embeddings."""

    user_id: str
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    created_at: Optional[str] = None

    def add_embedding(self, embedding: np.ndarray):
        """Add a new embedding to the profile."""
        self.embeddings.append(embedding)

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Get mean embedding across all stored embeddings."""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)


class FaceRecognizer:
    """Face detection and recognition using InsightFace.

    Usage:
        recognizer = FaceRecognizer()
        recognizer.load()

        # Enroll a user
        recognizer.enroll("user1", "John", frames)

        # Recognize faces
        result = recognizer.detect(frame)
        for face in result.faces:
            user_id, confidence = recognizer.identify(face)
    """

    def __init__(
        self,
        model: str = "buffalo_l",
        threshold: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
        device: str = "auto",
        embeddings_path: Optional[str | Path] = None,
    ):
        """Initialize recognizer.

        Args:
            model: InsightFace model pack name
            threshold: Cosine similarity threshold for matching
            det_size: Face detection input size
            device: Device (auto, cuda, cpu)
            embeddings_path: Path to store user embeddings
        """
        self.model_name = model
        self.threshold = threshold
        self.det_size = det_size
        self.device = device
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None

        self._app = None
        self._loaded = False
        self._users: dict[str, UserProfile] = {}

    def load(self):
        """Load InsightFace model."""
        if self._loaded:
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Run: pip install insightface"
            )

        logger.info(f"Loading InsightFace model: {self.model_name}")

        # Determine providers
        if self.device == "auto":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)

        self._loaded = True
        logger.info("InsightFace model loaded")

        # Load stored embeddings
        if self.embeddings_path and self.embeddings_path.exists():
            self._load_embeddings()

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> RecognitionResult:
        """Detect and analyze faces in frame.

        Args:
            frame: BGR image (OpenCV format)
            frame_id: Optional frame identifier

        Returns:
            RecognitionResult with all detected faces
        """
        if not self._loaded:
            self.load()

        import time

        start = time.time()

        # InsightFace expects RGB
        rgb_frame = frame[:, :, ::-1]
        faces_raw = self._app.get(rgb_frame)

        inference_time = (time.time() - start) * 1000

        faces = []
        for face in faces_raw:
            bbox = face.bbox.astype(int)
            faces.append(
                Face(
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    confidence=float(face.det_score),
                    embedding=face.embedding,
                    landmarks=face.kps if hasattr(face, "kps") else None,
                    age=int(face.age) if hasattr(face, "age") else None,
                    gender="M" if hasattr(face, "gender") and face.gender == 1 else "F"
                    if hasattr(face, "gender")
                    else None,
                )
            )

        return RecognitionResult(
            faces=faces,
            inference_time_ms=inference_time,
            frame_id=frame_id,
        )

    def identify(self, face: Face) -> tuple[Optional[str], float]:
        """Identify a face against enrolled users.

        Args:
            face: Face with embedding

        Returns:
            Tuple of (user_id, confidence) or (None, 0.0) if no match
        """
        if face.embedding is None:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for user_id, profile in self._users.items():
            mean_embedding = profile.get_mean_embedding()
            if mean_embedding is None:
                continue

            # Cosine similarity
            score = self._cosine_similarity(face.embedding, mean_embedding)

            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = user_id

        return best_match, best_score

    def enroll(
        self,
        user_id: str,
        name: str,
        frames: list[np.ndarray],
        min_faces: int = 3,
    ) -> bool:
        """Enroll a new user from multiple frames.

        Args:
            user_id: Unique user identifier
            name: Display name
            frames: List of frames containing the user's face
            min_faces: Minimum faces required for enrollment

        Returns:
            True if enrollment successful
        """
        if not self._loaded:
            self.load()

        embeddings = []

        for frame in frames:
            result = self.detect(frame)
            if result.has_face:
                largest = result.get_largest()
                if largest and largest.embedding is not None:
                    embeddings.append(largest.embedding)

        if len(embeddings) < min_faces:
            logger.warning(
                f"Enrollment failed: only {len(embeddings)} faces found, "
                f"need at least {min_faces}"
            )
            return False

        profile = UserProfile(
            user_id=user_id,
            name=name,
            embeddings=embeddings,
        )
        self._users[user_id] = profile

        # Save embeddings
        if self.embeddings_path:
            self._save_embeddings()

        logger.info(f"Enrolled user {user_id} with {len(embeddings)} embeddings")
        return True

    def enroll_from_embedding(
        self,
        user_id: str,
        name: str,
        embedding: np.ndarray,
    ):
        """Enroll a user from a pre-computed embedding.

        Args:
            user_id: Unique user identifier
            name: Display name
            embedding: Face embedding vector
        """
        if user_id not in self._users:
            self._users[user_id] = UserProfile(user_id=user_id, name=name)

        self._users[user_id].add_embedding(embedding)

        if self.embeddings_path:
            self._save_embeddings()

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self._users.get(user_id)

    def remove_user(self, user_id: str) -> bool:
        """Remove a user from the system."""
        if user_id in self._users:
            del self._users[user_id]
            if self.embeddings_path:
                self._save_embeddings()
            return True
        return False

    def list_users(self) -> list[str]:
        """List all enrolled user IDs."""
        return list(self._users.keys())

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        a = a.flatten()
        b = b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _save_embeddings(self):
        """Save user embeddings to disk."""
        if not self.embeddings_path:
            return

        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        for user_id, profile in self._users.items():
            user_file = self.embeddings_path / f"{user_id}.npz"
            np.savez(
                user_file,
                embeddings=np.array(profile.embeddings),
                name=profile.name,
            )

        logger.info(f"Saved {len(self._users)} user embeddings")

    def _load_embeddings(self):
        """Load user embeddings from disk."""
        if not self.embeddings_path or not self.embeddings_path.exists():
            return

        for user_file in self.embeddings_path.glob("*.npz"):
            user_id = user_file.stem
            data = np.load(user_file, allow_pickle=True)

            profile = UserProfile(
                user_id=user_id,
                name=str(data["name"]),
                embeddings=list(data["embeddings"]),
            )
            self._users[user_id] = profile

        logger.info(f"Loaded {len(self._users)} user embeddings")

    def warmup(self, size: tuple[int, int] = (640, 480)):
        """Warmup model with dummy inference."""
        if not self._loaded:
            self.load()

        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("Recognizer warmup complete")

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self._loaded
