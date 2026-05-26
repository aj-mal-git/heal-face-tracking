"""
TrackerManager: wraps deep_sort_realtime for multi-face tracking.
Uses ArcFace embeddings (from FaceEngine) instead of DeepSORT's internal
ResNet embedder — this gives far superior re-identification.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from core.face_engine import FaceResult


@dataclass
class TrackedFace:
    track_id: int
    bbox: List[float]          # [x1, y1, x2, y2]
    embedding: Optional[np.ndarray]
    det_score: float
    is_confirmed: bool


class TrackerManager:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
    ):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=None,   # we supply ArcFace embeddings directly
            half=True,
        )

    def update(
        self,
        face_results: List[FaceResult],
        frame: np.ndarray,
        frame_idx: int,
    ) -> List[TrackedFace]:
        """
        Update tracker with new detections.
        Returns TrackedFace list with track IDs assigned.
        """
        if not face_results:
            self._tracker.update_tracks([], frame=frame)
            return []

        # Build DeepSORT detection list: ([x, y, w, h], confidence, class_id)
        # Embeddings are passed separately via embeds= when no internal embedder is used
        detections = []
        embeddings = []
        for fr in face_results:
            x1, y1, x2, y2 = fr.bbox
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], fr.det_score, "face"))
            embeddings.append(fr.embedding)

        tracks = self._tracker.update_tracks(detections, embeds=embeddings, frame=frame)

        # Match confirmed tracks back to original FaceResults to get embeddings
        tracked_faces = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            # Find the face result with best IoU overlap
            matched_face = self._match_to_detection(ltrb, face_results)
            tracked_faces.append(TrackedFace(
                track_id=track.track_id,
                bbox=ltrb.tolist() if hasattr(ltrb, "tolist") else list(ltrb),
                embedding=matched_face.embedding if matched_face else None,
                det_score=matched_face.det_score if matched_face else 0.0,
                is_confirmed=True,
            ))

        return tracked_faces

    def _match_to_detection(
        self, ltrb: np.ndarray, face_results: List[FaceResult]
    ) -> Optional[FaceResult]:
        """Find the FaceResult with highest IoU against a track's bbox."""
        if not face_results:
            return None
        best_iou = 0.0
        best_face = None
        tx1, ty1, tx2, ty2 = ltrb[:4]
        for fr in face_results:
            fx1, fy1, fx2, fy2 = fr.bbox
            iou = self._iou(tx1, ty1, tx2, ty2, fx1, fy1, fx2, fy2)
            if iou > best_iou:
                best_iou = iou
                best_face = fr
        return best_face if best_iou > 0.1 else None

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)
