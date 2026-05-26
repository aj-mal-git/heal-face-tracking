"""
StreamProcessor: main processing loop for a single CCTV camera.
Runs in a background thread. Exposes latest annotated frame + results
for the API to serve to the dashboard.
"""
from __future__ import annotations
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

from core.config import settings
from core.face_engine import FaceEngine
from core.faiss_store import FaissStore
from core.tracker import TrackerManager
from core.recognition import RecognitionPipeline
from core.alert_manager import AlertManager
from core.schemas import RecognitionResult

# Annotation colors
COLOR_KNOWN = (0, 255, 0)      # Green
COLOR_UNKNOWN = (0, 0, 255)    # Red
COLOR_TEXT_BG = (0, 0, 0)


class StreamProcessor:
    def __init__(
        self,
        stream_url: str,
        camera_id: str,
        face_engine: FaceEngine,
        faiss_store: FaissStore,
        db_session_factory,
    ):
        self._url = stream_url
        self._camera_id = camera_id
        self._face_engine = face_engine

        self._tracker = TrackerManager()
        self._recognizer = RecognitionPipeline(
            faiss_store, db_session_factory, settings.recognition_threshold
        )
        self._alerter = AlertManager(
            db_session_factory, settings.snapshot_dir, settings.unknown_alert_cooldown
        )
        self._event_logger = EventLogger(
            db_session_factory, settings.attendance_log_interval
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_results: List[RecognitionResult] = []
        self._frame_count = 0
        self._active_tracks = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"stream-{self._camera_id}")
        self._thread.start()
        print(f"[StreamProcessor] Started camera '{self._camera_id}' -> {self._url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print(f"[StreamProcessor] Stopped camera '{self._camera_id}'")

    def is_running(self) -> bool:
        return self._running

    def get_latest(self) -> Tuple[Optional[np.ndarray], List[RecognitionResult]]:
        with self._lock:
            return self._latest_frame, list(self._latest_results)

    def get_stats(self) -> Dict:
        return {
            "running": self._running,
            "camera_id": self._camera_id,
            "frame_count": self._frame_count,
            "active_tracks": self._active_tracks,
        }

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _loop(self):
        # Handle numeric webcam index
        url = int(self._url) if self._url.isdigit() else self._url
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print(f"[StreamProcessor] ERROR: Cannot open stream '{self._url}'")
            self._running = False
            return

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                print(f"[StreamProcessor] Stream read failed, retrying...")
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(url)
                continue

            frame_idx += 1
            self._frame_count = frame_idx

            # Always update the raw frame so dashboard has latest image
            if frame_idx % settings.frame_skip != 0:
                with self._lock:
                    if self._latest_frame is None:
                        self._latest_frame = frame
                continue

            # ── Full pipeline ──
            # Downscale to 720p max before inference — reduces pixel count
            # while keeping enough detail for face detection at normal distances.
            h, w = frame.shape[:2]
            if h > 720:
                scale = 720 / h
                small = cv2.resize(frame, (int(w * scale), 720), interpolation=cv2.INTER_LINEAR)
            else:
                small = frame

            face_results = self._face_engine.detect_and_embed(small)

            # Scale bboxes back to original frame coords for correct annotation
            if h > 720:
                sx, sy = w / small.shape[1], h / small.shape[0]
                for fr in face_results:
                    fr.bbox = [
                        int(fr.bbox[0] * sx), int(fr.bbox[1] * sy),
                        int(fr.bbox[2] * sx), int(fr.bbox[3] * sy),
                    ]

            tracked_faces = self._tracker.update(face_results, frame, frame_idx)
            recognition_results = self._recognizer.process(tracked_faces, frame_idx)

            self._active_tracks = len(recognition_results)

            # Log attendance events
            for r in recognition_results:
                self._event_logger.maybe_log(r, self._camera_id)

            # Check for unknown alerts
            for r in recognition_results:
                self._alerter.maybe_alert(r, frame, self._camera_id)

            # Annotate frame
            annotated = self._annotate(frame, recognition_results)

            with self._lock:
                self._latest_frame = annotated
                self._latest_results = recognition_results

        cap.release()

    # ─── Annotation ──────────────────────────────────────────────────────────

    def _annotate(
        self, frame: np.ndarray, results: List[RecognitionResult]
    ) -> np.ndarray:
        out = frame.copy()

        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            color = COLOR_UNKNOWN if r.is_unknown else COLOR_KNOWN

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{r.name} ({r.confidence:.2f})"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(y1 - 5, lh + 5)
            cv2.rectangle(
                out,
                (x1, label_y - lh - baseline),
                (x1 + lw, label_y + baseline),
                color, cv2.FILLED,
            )
            cv2.putText(
                out, label, (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            # Track ID (small, top-right of box)
            cv2.putText(
                out, f"T{r.track_id}", (x2 - 30, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # Timestamp overlay
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            out, ts, (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.putText(
            out, f"CAM: {self._camera_id}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return out


# ─── Event Logger ─────────────────────────────────────────────────────────────

class EventLogger:
    """Logs recognized/unknown events to DB, throttled per track_id."""

    def __init__(self, db_session_factory, interval_seconds: int):
        self._db_factory = db_session_factory
        self._interval = interval_seconds
        self._last_logged: Dict[int, float] = {}  # track_id → epoch

    def maybe_log(self, result: RecognitionResult, camera_id: str):
        now = time.time()
        last = self._last_logged.get(result.track_id, 0.0)
        if now - last < self._interval:
            return
        self._last_logged[result.track_id] = now
        self._write(result, camera_id)

    def _write(self, result: RecognitionResult, camera_id: str):
        try:
            from core.models import Event
            with self._db_factory() as session:
                event = Event(
                    employee_id=result.employee_id,
                    track_id=result.track_id,
                    camera_id=camera_id,
                    confidence=result.confidence,
                    event_type="recognized" if not result.is_unknown else "unknown",
                )
                session.add(event)
                session.commit()
        except Exception as e:
            print(f"[EventLogger] DB write failed: {e}")
