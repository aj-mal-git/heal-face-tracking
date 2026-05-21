"""
Standalone stream test — run the full pipeline without the API server.
Shows an OpenCV window with face detection overlays.
Run: python scripts/test_stream.py --source 0
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2


def test_stream(source):
    from core.database import init_db, SessionLocal
    from core.face_engine import get_face_engine
    from core.faiss_store import get_faiss_store
    from core.stream_processor import StreamProcessor
    from contextlib import contextmanager

    init_db()

    @contextmanager
    def db_ctx():
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

    print(f"Loading face engine (first run downloads model ~300MB)...")
    face_engine = get_face_engine()
    faiss_store = get_faiss_store()
    print(f"Faiss index: {faiss_store.count()} embeddings")

    processor = StreamProcessor(
        stream_url=str(source),
        camera_id="test",
        face_engine=face_engine,
        faiss_store=faiss_store,
        db_session_factory=db_ctx,
    )
    processor.start()

    print("Press 'q' to quit")
    try:
        while True:
            frame, results = processor.get_latest()
            if frame is not None:
                cv2.imshow("HEAL Face Recognition — Test", frame)
                if results:
                    for r in results:
                        status = r.name if not r.is_unknown else "UNKNOWN"
                        print(f"\r  Track {r.track_id}: {status} ({r.confidence:.2f})    ", end="")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        print("\nTest stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    test_stream(source)
