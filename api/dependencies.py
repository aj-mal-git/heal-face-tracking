from core.database import get_db, SessionLocal
from core.face_engine import get_face_engine, FaceEngine
from core.faiss_store import get_faiss_store, FaissStore
from contextlib import contextmanager


def db_dependency():
    """FastAPI Depends: yields a DB session."""
    return get_db()


def get_engine() -> FaceEngine:
    return get_face_engine()


def get_store() -> FaissStore:
    return get_faiss_store()


@contextmanager
def db_context():
    """Context manager for use outside FastAPI request cycle (e.g. stream processor)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
