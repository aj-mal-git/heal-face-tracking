"""
Multi-camera stream management endpoints.

Architecture (from spec):
  Camera 1 RTSP ──┐
  Camera 2 RTSP ──┼──► POST /api/streams/start  (camera_id + url)
  Camera N RTSP ──┘

POST /api/streams/start              — start a camera stream
POST /api/streams/stop/{camera_id}   — stop a specific camera
POST /api/streams/stop-all           — stop all cameras
GET  /api/streams/                   — list all active streams
GET  /api/streams/status/{camera_id} — status of a specific camera
GET  /api/streams/frame/{camera_id}  — latest annotated JPEG frame
GET  /api/streams/frame-b64/{camera_id} — latest frame as base64 JSON
"""
import base64
import io

import cv2
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from core.schemas import StreamConfig, StreamStatus

router = APIRouter()


def _manager(request: Request):
    return request.app.state.stream_manager


# ─── Start ────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StreamStatus)
def start_stream(config: StreamConfig, request: Request):
    """
    Start processing a camera stream.
    - url: RTSP URL (e.g. rtsp://192.168.1.10:554/stream) or '0' for webcam
    - camera_id: unique name for this camera (e.g. 'cam1', 'entrance', 'lobby')
    Multiple cameras can run simultaneously — each gets its own processor thread.
    """
    manager = _manager(request)
    manager.start(camera_id=config.camera_id, url=config.url)
    return StreamStatus(
        running=True,
        camera_id=config.camera_id,
        frame_count=0,
        active_tracks=0,
    )


# ─── Stop ─────────────────────────────────────────────────────────────────────

@router.post("/stop/{camera_id}")
def stop_stream(camera_id: str, request: Request):
    """Stop a specific camera stream by camera_id."""
    manager = _manager(request)
    stopped = manager.stop(camera_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"No active stream for camera '{camera_id}'")
    return {"message": f"Stream '{camera_id}' stopped"}


@router.post("/stop")
def stop_stream_legacy(request: Request):
    """Legacy single-stream stop — stops first active stream (backwards compat)."""
    manager = _manager(request)
    active = manager.list_active()
    if not active:
        raise HTTPException(status_code=404, detail="No active streams")
    camera_id = active[0]["camera_id"]
    manager.stop(camera_id)
    return {"message": f"Stream '{camera_id}' stopped"}


@router.post("/stop-all")
def stop_all_streams(request: Request):
    """Stop all active camera streams."""
    manager = _manager(request)
    count = manager.count()
    manager.stop_all()
    return {"message": f"Stopped {count} stream(s)"}


# ─── List / Status ────────────────────────────────────────────────────────────

@router.get("/")
def list_streams(request: Request):
    """List all active camera streams with their stats."""
    manager = _manager(request)
    return {
        "active_cameras": manager.count(),
        "streams": manager.list_active(),
    }


@router.get("/status/{camera_id}", response_model=StreamStatus)
def stream_status(camera_id: str, request: Request):
    """Get status of a specific camera stream."""
    manager = _manager(request)
    processor = manager.get(camera_id)
    if not processor:
        return StreamStatus(running=False, camera_id=camera_id, frame_count=0, active_tracks=0)
    return StreamStatus(**processor.get_stats())


@router.get("/status", response_model=StreamStatus)
def stream_status_legacy(request: Request):
    """Legacy single-stream status (backwards compat with dashboard)."""
    manager = _manager(request)
    active = manager.list_active()
    if not active:
        return StreamStatus(running=False, camera_id=None, frame_count=0, active_tracks=0)
    s = active[0]
    return StreamStatus(
        running=s["running"],
        camera_id=s["camera_id"],
        frame_count=s["frame_count"],
        active_tracks=s["active_tracks"],
    )


# ─── Frame endpoints ──────────────────────────────────────────────────────────

@router.get("/frame/{camera_id}")
def latest_frame(camera_id: str, request: Request):
    """Return latest annotated JPEG frame for a specific camera."""
    manager = _manager(request)
    processor = manager.get(camera_id)
    if not processor:
        raise HTTPException(status_code=404, detail=f"No active stream for camera '{camera_id}'")

    frame, _ = processor.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/frame")
def latest_frame_legacy(request: Request):
    """Legacy single-camera frame endpoint (backwards compat with dashboard)."""
    manager = _manager(request)
    active = manager.list_active()
    if not active:
        raise HTTPException(status_code=404, detail="No active stream")
    processor = manager.get(active[0]["camera_id"])
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


@router.get("/frame-b64/{camera_id}")
def latest_frame_b64(camera_id: str, request: Request):
    """Return latest frame + recognition results as base64 JSON for a specific camera."""
    manager = _manager(request)
    processor = manager.get(camera_id)
    if not processor:
        raise HTTPException(status_code=404, detail=f"No active stream for camera '{camera_id}'")

    frame, results = processor.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_b64 = base64.b64encode(buffer.tobytes()).decode()

    return {
        "camera_id": camera_id,
        "frame_b64": frame_b64,
        "results": [r.model_dump() for r in results],
        "active_tracks": len(results),
    }


@router.get("/frame-b64")
def latest_frame_b64_legacy(request: Request):
    """Legacy single-camera frame-b64 endpoint (backwards compat with dashboard)."""
    manager = _manager(request)
    active = manager.list_active()
    if not active:
        raise HTTPException(status_code=404, detail="No active stream")
    camera_id = active[0]["camera_id"]
    processor = manager.get(camera_id)
    if not processor:
        raise HTTPException(status_code=404, detail="No active stream")

    frame, results = processor.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_b64 = base64.b64encode(buffer.tobytes()).decode()

    return {
        "camera_id": camera_id,
        "frame_b64": frame_b64,
        "results": [r.model_dump() for r in results],
        "active_tracks": len(results),
    }
