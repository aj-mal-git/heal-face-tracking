"""
APIClient: centralized HTTP client for all FastAPI calls from Streamlit.
All methods are synchronous (Streamlit runs synchronously).
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import date

import httpx


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=15.0)

    # ─── Employees ───────────────────────────────────────────────────────────

    def get_employees(self, active_only: bool = True) -> List[Dict]:
        try:
            r = self._client.get(f"{self.base}/api/employees/", params={"active_only": active_only})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return []

    def get_employee(self, employee_id: int) -> Optional[Dict]:
        try:
            r = self._client.get(f"{self.base}/api/employees/{employee_id}")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def enroll_employee(
        self, name: str, department: str, email: str, photo_bytes: bytes, filename: str
    ) -> Dict:
        try:
            r = self._client.post(
                f"{self.base}/api/employees/enroll",
                data={"name": name, "department": department, "email": email},
                files={"photo": (filename, photo_bytes, "image/jpeg")},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    def deactivate_employee(self, employee_id: int) -> Dict:
        try:
            r = self._client.delete(f"{self.base}/api/employees/{employee_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def re_enroll(self, employee_id: int, photo_bytes: bytes, filename: str) -> Dict:
        try:
            r = self._client.post(
                f"{self.base}/api/employees/{employee_id}/re-enroll",
                files={"photo": (filename, photo_bytes, "image/jpeg")},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception as e:
            return {"error": str(e)}

    # ─── Streams ─────────────────────────────────────────────────────────────

    def start_stream(self, url: str, camera_id: str = "main") -> Dict:
        try:
            r = self._client.post(
                f"{self.base}/api/streams/start",
                json={"url": url, "camera_id": camera_id},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def stop_stream(self) -> Dict:
        try:
            r = self._client.post(f"{self.base}/api/streams/stop")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_stream_status(self) -> Dict:
        try:
            r = self._client.get(f"{self.base}/api/streams/status")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"running": False, "camera_id": None, "frame_count": 0, "active_tracks": 0}

    def get_latest_frame(self) -> Optional[bytes]:
        """Returns raw JPEG bytes of the latest annotated frame."""
        try:
            r = self._client.get(f"{self.base}/api/streams/frame", timeout=3.0)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        return None

    def get_frame_b64(self) -> Optional[Dict]:
        try:
            r = self._client.get(f"{self.base}/api/streams/frame-b64", timeout=3.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # ─── Events ──────────────────────────────────────────────────────────────

    def get_events(
        self,
        employee_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        params = {"limit": limit}
        if employee_id:
            params["employee_id"] = employee_id
        if camera_id:
            params["camera_id"] = camera_id
        if event_type:
            params["event_type"] = event_type
        if from_dt:
            params["from_dt"] = from_dt
        if to_dt:
            params["to_dt"] = to_dt
        try:
            r = self._client.get(f"{self.base}/api/events/", params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def get_attendance(self, target_date: Optional[str] = None) -> List[Dict]:
        params = {}
        if target_date:
            params["date"] = target_date
        try:
            r = self._client.get(f"{self.base}/api/events/attendance", params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    # ─── Alerts ──────────────────────────────────────────────────────────────

    def get_alerts(self, resolved: Optional[bool] = False, limit: int = 100) -> List[Dict]:
        params = {"limit": limit}
        if resolved is not None:
            params["resolved"] = resolved
        try:
            r = self._client.get(f"{self.base}/api/alerts/", params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def get_alert_count(self, resolved: bool = False) -> int:
        try:
            r = self._client.get(f"{self.base}/api/alerts/count", params={"resolved": resolved})
            r.raise_for_status()
            return r.json().get("count", 0)
        except Exception:
            return 0

    def resolve_alert(self, alert_id: int, notes: str = "") -> Dict:
        try:
            params = {}
            if notes:
                params["notes"] = notes
            r = self._client.post(f"{self.base}/api/alerts/{alert_id}/resolve", params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_snapshot_url(self, alert: Dict) -> str:
        return f"{self.base}/api/alerts/{alert['id']}/snapshot"

    # ─── Health ──────────────────────────────────────────────────────────────

    def health(self) -> Dict:
        try:
            r = self._client.get(f"{self.base}/health", timeout=2.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "unreachable"}
