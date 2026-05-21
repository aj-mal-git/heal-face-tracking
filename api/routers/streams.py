"""
Stream management endpoints.
POST /api/streams/start   — start processing a camera stream
POST /api/streams/stop    — stop the current stream
GET  /api/streams/status  — get current stream status
GET  /api/streams/frame   — get latest annotated JPEG frame
"""
import base64
import io
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from core.schemas import StreamConfig, StreamStatus
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store
from api.dependencies import db_context

router = APIRouter()


def _get_processor(request: Request):
    return getattr(request.app.state, "stream_processor", None)


@router.post("/start", response_model=StreamStatus)
def start_stream(config: StreamConfig, request: Request):
    from core.stream_processor import StreamProcessor

    existing = _get_processor(request)
    if existing and existing.is_running():
        existing.stop()

    processor = StreamProcessor(
        stream_url=config.url,
        camera_id=config.camera_id,
        face_engine=get_face_engine(),
        faiss_store=get_faiss_store(),
        db_session_factory=db_context,
    )
    processor.start()
    request.app.state.stream_processor = processor

    return StreamStatus(
        running=True,
        camera_id=config.camera_id,
        frame_count=0,
        active_tracks=0,
    )


@router.post("/stop")
def stop_stream(request: Request):
    processor = _get_processor(request)
    if not processor:
        raise HTTPException(status_code=404, detail="No active stream")
    processor.stop()
    request.app.state.stream_processor = None
    return {"message": "Stream stopped"}


@router.get("/status", response_model=StreamStatus)
def stream_status(request: Request):
    processor = _get_processor(request)
    if not processor:
        return StreamStatus(running=False, camera_id=None, frame_count=0, active_tracks=0)
    stats = processor.get_stats()
    return StreamStatus(**stats)


@router.get("/frame")
def latest_frame(request: Request):
    """Return the latest annotated frame as a JPEG image."""
    processor = _get_processor(request)
    if not processor:
        raise HTTPException(status_code=404, detail="No active stream")

    frame, _ = processor.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/frame-b64")
def latest_frame_b64(request: Request):
    """Return the latest frame as base64 JSON (for WebSocket-style polling)."""
    processor = _get_processor(request)
    if not processor:
        raise HTTPException(status_code=404, detail="No active stream")

    frame, results = processor.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_b64 = base64.b64encode(buffer.tobytes()).decode()

    return {
        "frame_b64": frame_b64,
        "results": [r.model_dump() for r in results],
        "active_tracks": len(results),
    }
