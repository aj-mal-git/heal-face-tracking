"""
RecognitionPipeline: converts TrackedFace list → RecognitionResult list.
Caches results per track_id to avoid querying Faiss every frame.
Re-queries Faiss every RERECOGNIZE_EVERY_N_FRAMES for updated accuracy.
"""
from __future__ import annotations
from typing import Optional, List, Dict

from core.faiss_store import FaissStore
from core.tracker import TrackedFace
from core.schemas import RecognitionResult

# Re-query Faiss for a confirmed track every N frames
RERECOGNIZE_EVERY_N_FRAMES = 15


class RecognitionPipeline:
    def __init__(
        self,
        faiss_store: FaissStore,
        db_session_factory,
        threshold: float,
    ):
        self._faiss = faiss_store
        self._db_factory = db_session_factory
        self._threshold = threshold
        self._cache: Dict[int, RecognitionResult] = {}
        self._frame_counter: Dict[int, int] = {}

    def process(
        self, tracked_faces: List[TrackedFace], frame_idx: int
    ) -> List[RecognitionResult]:
        results = []
        active_track_ids = set()

        for tf in tracked_faces:
            if not tf.is_confirmed:
                continue
            active_track_ids.add(tf.track_id)

            # Use cached result if fresh enough
            frames_since = frame_idx - self._frame_counter.get(tf.track_id, -9999)
            if tf.track_id in self._cache and frames_since < RERECOGNIZE_EVERY_N_FRAMES:
                # Update bbox to current position
                cached = self._cache[tf.track_id]
                updated = RecognitionResult(
                    track_id=cached.track_id,
                    bbox=tf.bbox,
                    employee_id=cached.employee_id,
                    name=cached.name,
                    confidence=cached.confidence,
                    is_unknown=cached.is_unknown,
                )
                results.append(updated)
                continue

            if tf.embedding is None:
                continue

            result = self._recognize(tf)
            self._cache[tf.track_id] = result
            self._frame_counter[tf.track_id] = frame_idx
            results.append(result)

        # Evict stale cache entries for tracks no longer active
        dead_ids = set(self._cache.keys()) - active_track_ids
        for dead_id in dead_ids:
            self._cache.pop(dead_id, None)
            self._frame_counter.pop(dead_id, None)

        return results

    def _recognize(self, tf: TrackedFace) -> RecognitionResult:
        matches = self._faiss.search(tf.embedding, top_k=1)

        if matches and matches[0][1] >= self._threshold:
            employee_id, similarity = matches[0]
            name = self._get_employee_name(employee_id)
            return RecognitionResult(
                track_id=tf.track_id,
                bbox=tf.bbox,
                employee_id=employee_id,
                name=name or f"Employee #{employee_id}",
                confidence=similarity,
                is_unknown=False,
            )
        else:
            return RecognitionResult(
                track_id=tf.track_id,
                bbox=tf.bbox,
                employee_id=None,
                name="Unknown",
                confidence=matches[0][1] if matches else 0.0,
                is_unknown=True,
            )

    def _get_employee_name(self, employee_id: int) -> Optional[str]:
        try:
            with self._db_factory() as session:
                from core.models import Employee
                emp = session.get(Employee, employee_id)
                return emp.name if emp else None
        except Exception:
            return None
