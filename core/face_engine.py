"""
FaceEngine: wraps InsightFace (RetinaFace detector + ArcFace recognizer).
Provides detect_and_embed() for frames and embed_from_bytes() for enrollment.
Thread-safe via internal lock.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np

_engine_instance: Optional["FaceEngine"] = None
_engine_lock = threading.Lock()


@dataclass
class FaceResult:
    bbox: List[int]           # [x1, y1, x2, y2]
    embedding: np.ndarray     # 512-d L2-normalized ArcFace vector
    det_score: float
    kps: Optional[List] = field(default=None)  # 5 facial keypoints


class FaceEngine:
    """
    Singleton wrapper around InsightFace FaceAnalysis.
    Uses RetinaFace for detection and ArcFace (R100) for recognition.
    """

    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis

        # buffalo_sc = SCRFD detector + ArcFace recognizer, CPU-optimised.
        # Per arch spec: handles 50 people / 6 cameras on CPU without GPU.
        # ~15ms/face on CPU vs buffalo_l which loads 5 models unnecessarily.
        self._app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # 320×320 is sufficient for CCTV face detection and 4× faster than 640×640
        self._app.prepare(ctx_id=0, det_size=(320, 320))
        self._lock = threading.Lock()
        print("[FaceEngine] InsightFace buffalo_l loaded successfully.")

    def detect_and_embed(self, bgr_frame: np.ndarray) -> List[FaceResult]:
        """Detect all faces in a frame and return embeddings."""
        if bgr_frame is None or bgr_frame.size == 0:
            return []
        with self._lock:
            faces = self._app.get(bgr_frame)
        results = []
        for face in faces:
            if face.det_score < 0.5:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:
                continue  # skip tiny faces
            results.append(FaceResult(
                bbox=[x1, y1, x2, y2],
                embedding=face.normed_embedding,  # already L2-normalized
                det_score=float(face.det_score),
                kps=face.kps.tolist() if face.kps is not None else None,
            ))
        return results

    def embed_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract face embedding from raw image bytes (for enrollment API)."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return self._embed_image(img)

    def embed_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding from file path (for bulk enrollment)."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self._embed_image(img)

    def _embed_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        results = self.detect_and_embed(img)
        if not results:
            return None
        # Pick the highest-confidence face
        best = max(results, key=lambda r: r.det_score)
        return best.embedding


def get_face_engine() -> FaceEngine:
    """Return singleton FaceEngine, initializing on first call (~3-5s)."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = FaceEngine()
    return _engine_instance
