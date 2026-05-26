"""
StreamManager: manages multiple simultaneous RTSP camera streams.
Architecture: Camera N RTSP → StreamProcessor (per camera) → FAISS Matcher
Each camera runs its own background thread with its own tracker instance.
The FaceEngine and FaissStore are shared singletons (thread-safe).
"""
from __future__ import annotations
import threading
from typing import Dict, Optional, List

from core.stream_processor import StreamProcessor
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store
from api.dependencies import db_context


class StreamManager:
    """
    Manages N concurrent camera streams as per the architecture:
      Camera 1 RTSP ──┐
      Camera 2 RTSP ──┼──► StreamManager ──► shared FaceEngine + FAISS
      Camera N RTSP ──┘
    """

    def __init__(self):
        self._streams: Dict[str, StreamProcessor] = {}
        self._lock = threading.Lock()

    # ─── Start / Stop ─────────────────────────────────────────────────────────

    def start(self, camera_id: str, url: str) -> StreamProcessor:
        """Start a new stream for camera_id. Replaces any existing stream for that ID."""
        with self._lock:
            # Stop existing stream for this camera if running
            if camera_id in self._streams:
                old = self._streams[camera_id]
                if old.is_running():
                    old.stop()

            processor = StreamProcessor(
                stream_url=url,
                camera_id=camera_id,
                face_engine=get_face_engine(),    # shared singleton
                faiss_store=get_faiss_store(),    # shared singleton
                db_session_factory=db_context,
            )
            processor.start()
            self._streams[camera_id] = processor
            print(f"[StreamManager] Started stream: camera_id='{camera_id}' url='{url}'")
            return processor

    def stop(self, camera_id: str) -> bool:
        """Stop a specific camera stream. Returns True if it was running."""
        with self._lock:
            processor = self._streams.get(camera_id)
            if not processor:
                return False
            processor.stop()
            del self._streams[camera_id]
            print(f"[StreamManager] Stopped stream: camera_id='{camera_id}'")
            return True

    def stop_all(self):
        """Stop all active streams (called on API shutdown)."""
        with self._lock:
            for camera_id, processor in list(self._streams.items()):
                if processor.is_running():
                    processor.stop()
                    print(f"[StreamManager] Stopped stream: camera_id='{camera_id}'")
            self._streams.clear()

    # ─── Query ────────────────────────────────────────────────────────────────

    def get(self, camera_id: str) -> Optional[StreamProcessor]:
        with self._lock:
            return self._streams.get(camera_id)

    def list_active(self) -> List[Dict]:
        """Return stats for all active streams."""
        with self._lock:
            return [
                {"camera_id": cid, **proc.get_stats()}
                for cid, proc in self._streams.items()
                if proc.is_running()
            ]

    def count(self) -> int:
        with self._lock:
            return sum(1 for p in self._streams.values() if p.is_running())

    def is_running(self, camera_id: str) -> bool:
        with self._lock:
            proc = self._streams.get(camera_id)
            return proc is not None and proc.is_running()
