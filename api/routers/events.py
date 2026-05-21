"""
Event query endpoints.
GET /api/events/              — list events with filters
GET /api/events/attendance    — daily attendance summary per employee
"""
from datetime import datetime, date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from core.database import get_db
from core.models import Event, Employee
from core.schemas import EventRead, AttendanceSummary

router = APIRouter()


@router.get("/", response_model=List[EventRead])
def list_events(
    employee_id: Optional[int] = None,
    camera_id: Optional[str] = None,
    event_type: Optional[str] = None,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    query = db.query(Event)

    if employee_id:
        query = query.filter(Event.employee_id == employee_id)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if event_type:
        query = query.filter(Event.event_type == event_type)
    if from_dt:
        query = query.filter(Event.timestamp >= from_dt)
    if to_dt:
        query = query.filter(Event.timestamp <= to_dt)

    events = query.order_by(Event.timestamp.desc()).offset(skip).limit(limit).all()

    # Enrich with employee names
    results = []
    for ev in events:
        ev_dict = {
            "id": ev.id,
            "employee_id": ev.employee_id,
            "employee_name": ev.employee.name if ev.employee else None,
            "track_id": ev.track_id,
            "camera_id": ev.camera_id,
            "confidence": ev.confidence,
            "event_type": ev.event_type,
            "timestamp": ev.timestamp,
        }
        results.append(EventRead(**ev_dict))
    return results


@router.get("/attendance", response_model=List[AttendanceSummary])
def attendance_summary(
    target_date: Optional[date] = Query(default=None, alias="date"),
    db: Session = Depends(get_db),
):
    """Per-employee attendance summary for a given date (default: today)."""
    if target_date is None:
        target_date = date.today()

    from_dt = datetime.combine(target_date, datetime.min.time())
    to_dt = datetime.combine(target_date, datetime.max.time())

    rows = (
        db.query(
            Event.employee_id,
            Employee.name,
            Employee.department,
            func.min(Event.timestamp).label("first_seen"),
            func.max(Event.timestamp).label("last_seen"),
            func.count(Event.id).label("appearances"),
        )
        .join(Employee, Event.employee_id == Employee.id)
        .filter(Event.timestamp >= from_dt)
        .filter(Event.timestamp <= to_dt)
        .filter(Event.employee_id.isnot(None))
        .group_by(Event.employee_id)
        .all()
    )

    return [
        AttendanceSummary(
            employee_id=r.employee_id,
            name=r.name,
            department=r.department,
            first_seen=r.first_seen,
            last_seen=r.last_seen,
            appearances=r.appearances,
        )
        for r in rows
    ]
