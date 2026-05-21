from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


# ─── Employee ────────────────────────────────────────────────────────────────

class EmployeeCreate(BaseModel):
    name: str
    department: Optional[str] = None
    email: Optional[str] = None


class EmployeeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    department: Optional[str]
    email: Optional[str]
    photo_path: Optional[str]
    enrolled_at: datetime
    is_active: bool


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None


# ─── Event ────────────────────────────────────────────────────────────────────

class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_id: Optional[int]
    employee_name: Optional[str] = None  # populated manually from join
    track_id: int
    camera_id: str
    confidence: Optional[float]
    event_type: str
    timestamp: datetime


# ─── Alert ───────────────────────────────────────────────────────────────────

class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_id: Optional[int]
    employee_id: Optional[int]
    snapshot_path: Optional[str]
    alert_type: str
    notes: Optional[str]
    resolved: bool
    created_at: datetime


# ─── Stream ──────────────────────────────────────────────────────────────────

class StreamConfig(BaseModel):
    url: str = "0"
    camera_id: str = "main"


class StreamStatus(BaseModel):
    running: bool
    camera_id: Optional[str]
    frame_count: int
    active_tracks: int


# ─── Recognition Results (live) ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class RecognitionResult(BaseModel):
    track_id: int
    bbox: List[float]          # [x1, y1, x2, y2]
    employee_id: Optional[int]
    name: str
    confidence: float
    is_unknown: bool


class LiveFrame(BaseModel):
    frame_b64: str             # base64-encoded JPEG
    results: List[RecognitionResult]
    timestamp: str


# ─── Attendance Summary ───────────────────────────────────────────────────────

class AttendanceSummary(BaseModel):
    employee_id: int
    name: str
    department: Optional[str]
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    appearances: int
