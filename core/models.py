from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from core.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    department = Column(String(64), nullable=True)
    email = Column(String(128), unique=True, nullable=True)
    photo_path = Column(String(256), nullable=True)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    events = relationship("Event", back_populates="employee")
    alerts = relationship("Alert", back_populates="employee")

    def __repr__(self):
        return f"<Employee id={self.id} name={self.name}>"


class Event(Base):
    """Every recognition result logged (throttled to avoid DB flooding)."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)  # NULL = unknown
    track_id = Column(Integer, nullable=False)
    camera_id = Column(String(64), default="main")
    confidence = Column(Float, nullable=True)
    event_type = Column(String(32), nullable=False)  # recognized | unknown | entry | exit
    timestamp = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="events")
    alert = relationship("Alert", back_populates="event", uselist=False)

    def __repr__(self):
        return f"<Event id={self.id} type={self.event_type} emp={self.employee_id}>"


class Alert(Base):
    """Raised when an unknown person is detected."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    snapshot_path = Column(String(256), nullable=True)
    alert_type = Column(String(32), nullable=False)  # unknown_person | low_confidence
    notes = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("Event", back_populates="alert")
    employee = relationship("Employee", back_populates="alerts")

    def __repr__(self):
        return f"<Alert id={self.id} type={self.alert_type} resolved={self.resolved}>"
