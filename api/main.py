"""
HEAL Face Recognition System — FastAPI Application

Architecture (from spec):
  CCTV Cameras (RTSP) → StreamManager → FaceEngine (SCRFD + ArcFace) → FAISS Matcher
  Web Frontend (Streamlit) ↔ FastAPI Backend ↔ Storage (FAISS + SQLite/MySQL)

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.database import init_db
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store
from core.stream_manager import StreamManager
from api.routers import employees, streams, events, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("[HEAL] Initializing database...")
    init_db()

    print("[HEAL] Loading face recognition model (buffalo_sc: SCRFD + ArcFace)...")
    get_face_engine()   # warms up InsightFace — loads ONNX models into CUDA/CPU

    print("[HEAL] Loading FAISS index...")
    store = get_faiss_store()
    print(f"[HEAL] FAISS index has {store.count()} enrolled embeddings.")

    # Multi-camera stream manager (replaces single stream_processor)
    app.state.stream_manager = StreamManager()

    print(f"[HEAL] API ready at http://{settings.api_host}:{settings.api_port}")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[HEAL] Stopping all camera streams...")
    app.state.stream_manager.stop_all()
    print("[HEAL] Shutdown complete.")


app = FastAPI(
    title="HEAL Face Recognition API",
    description=(
        "Multi-camera employee face detection, recognition & tracking. "
        "Architecture: CCTV RTSP → SCRFD detector → ArcFace embedder → FAISS matcher."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard (different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve snapshot face-crop images as static files
os.makedirs(settings.snapshot_dir, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=settings.snapshot_dir), name="snapshots")

# Routers
app.include_router(employees.router, prefix="/api/employees", tags=["Employees"])
app.include_router(streams.router,   prefix="/api/streams",   tags=["Streams"])
app.include_router(events.router,    prefix="/api/events",    tags=["Events"])
app.include_router(alerts.router,    prefix="/api/alerts",    tags=["Alerts"])


@app.get("/")
def root():
    return {
        "service": "HEAL Face Recognition System",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "running",
        "architecture": "SCRFD + ArcFace (buffalo_sc) + FAISS",
    }


@app.get("/health")
def health(request: "Request"):  # noqa: F821
    from fastapi import Request
    store = get_faiss_store()
    manager = request.app.state.stream_manager
    return {
        "status": "ok",
        "enrolled_faces": store.count(),
        "active_cameras": manager.count(),
        "streams": manager.list_active(),
    }
