"""
AlertManager: detects unknown persons and saves alerts with snapshots.
Applies a cooldown per track_id to avoid duplicate alerts.
"""
from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np

from core.config import settings
from core.schemas import RecognitionResult


class AlertManager:
    def __init__(self, db_session_factory, snapshot_dir: str, cooldown_seconds: int):
        self._db_factory = db_session_factory
        self._snapshot_dir = snapshot_dir
        self._cooldown = cooldown_seconds
        self._last_alert_time: Dict[int, float] = {}  # track_id → epoch time

    def maybe_alert(
        self,
        result: RecognitionResult,
        frame: np.ndarray,
        camera_id: str,
    ) -> Optional[int]:
        """
        If result is unknown and cooldown has passed, save snapshot + write DB alert.
        Returns alert ID if alert was raised, else None.
        """
        if not result.is_unknown:
            return None

        now = time.time()
        last = self._last_alert_time.get(result.track_id, 0.0)
        if now - last < self._cooldown:
            return None  # still within cooldown window

        self._last_alert_time[result.track_id] = now
        snapshot_path = self._save_snapshot(frame, result, camera_id)
        alert_id = self._write_to_db(result, camera_id, snapshot_path)
        print(
            f"[AlertManager] Unknown person! Track={result.track_id} "
            f"Camera={camera_id} Alert={alert_id}"
        )
        return alert_id

    def _save_snapshot(
        self, frame: np.ndarray, result: RecognitionResult, camera_id: str
    ) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = os.path.join(self._snapshot_dir, date_str)
        os.makedirs(out_dir, exist_ok=True)

        annotated = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, "UNKNOWN",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        timestamp_str = datetime.now().strftime("%H%M%S")
        filename = f"track{result.track_id}_{camera_id}_{timestamp_str}.jpg"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, annotated)
        return path

    def _write_to_db(
        self, result: RecognitionResult, camera_id: str, snapshot_path: str
    ) -> Optional[int]:
        try:
            from core.models import Event, Alert
            with self._db_factory() as session:
                event = Event(
                    employee_id=None,
                    track_id=result.track_id,
                    camera_id=camera_id,
                    confidence=result.confidence,
                    event_type="unknown",
                )
                session.add(event)
                session.flush()  # get event.id before commit

                alert = Alert(
                    event_id=event.id,
                    employee_id=None,
                    snapshot_path=snapshot_path,
                    alert_type="unknown_person",
                )
                session.add(alert)
                session.commit()
                return alert.id
        except Exception as e:
            print(f"[AlertManager] DB write failed: {e}")
            return None
