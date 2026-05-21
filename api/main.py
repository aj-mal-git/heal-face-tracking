"""
HEAL Face Recognition System — FastAPI Application
Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.database import init_db
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store
from api.routers import employees, streams, events, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("[HEAL] Initializing database...")
    init_db()

    print("[HEAL] Loading face recognition model (first run may take ~30s for download)...")
    get_face_engine()  # warms up InsightFace — loads ONNX model into CUDA/CPU

    print("[HEAL] Loading Faiss index...")
    store = get_faiss_store()
    print(f"[HEAL] Faiss index has {store.count()} embeddings.")

    app.state.stream_processor = None

    print(f"[HEAL] API ready at http://{settings.api_host}:{settings.api_port}")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    processor = getattr(app.state, "stream_processor", None)
    if processor and processor.is_running():
        print("[HEAL] Stopping stream processor...")
        processor.stop()
    print("[HEAL] Shutdown complete.")


app = FastAPI(
    title="HEAL Face Recognition API",
    description="Employee face detection, recognition & tracking for HEAL company CCTV system",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow dashboard (Streamlit) to call the API from different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount snapshot images as static files
import os
os.makedirs(settings.snapshot_dir, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=settings.snapshot_dir), name="snapshots")

# Include routers
app.include_router(employees.router, prefix="/api/employees", tags=["Employees"])
app.include_router(streams.router, prefix="/api/streams", tags=["Streams"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])


@app.get("/")
def root():
    return {
        "service": "HEAL Face Recognition System",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
def health():
    store = get_faiss_store()
    processor = app.state.stream_processor
    return {
        "status": "ok",
        "enrolled_faces": store.count(),
        "stream_active": processor is not None and processor.is_running(),
    }
