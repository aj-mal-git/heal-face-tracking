"""
Alert endpoints.
GET  /api/alerts/            — list alerts
POST /api/alerts/{id}/resolve — mark alert as resolved
GET  /api/alerts/{id}/snapshot — serve snapshot image
"""
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import Alert
from core.schemas import AlertRead

router = APIRouter()


@router.get("/", response_model=List[AlertRead])
def list_alerts(
    resolved: Optional[bool] = None,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    query = db.query(Alert)
    if resolved is not None:
        query = query.filter(Alert.resolved == resolved)
    if from_dt:
        query = query.filter(Alert.created_at >= from_dt)
    if to_dt:
        query = query.filter(Alert.created_at <= to_dt)
    return query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/count")
def alert_count(resolved: bool = False, db: Session = Depends(get_db)):
    count = db.query(Alert).filter(Alert.resolved == resolved).count()
    return {"count": count, "resolved": resolved}


@router.post("/{alert_id}/resolve", response_model=AlertRead)
def resolve_alert(
    alert_id: int,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.resolved = True
    if notes:
        alert.notes = notes
    db.commit()
    db.refresh(alert)
    return alert


@router.get("/{alert_id}/snapshot")
def get_snapshot(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    if not alert.snapshot_path or not os.path.exists(alert.snapshot_path):
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    return FileResponse(alert.snapshot_path, media_type="image/jpeg")
