import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    stream_url: str = os.getenv("STREAM_URL", "0")
    recognition_threshold: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))
    unknown_alert_cooldown: int = int(os.getenv("UNKNOWN_ALERT_COOLDOWN_SECONDS", "30"))
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "data/snapshots")
    enrolled_photos_dir: str = os.getenv("ENROLLED_PHOTOS_DIR", "data/enrolled_photos")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/faiss/index.faiss")
    faiss_map_path: str = os.getenv("FAISS_MAP_PATH", "data/faiss/id_map.pkl")
    db_url: str = os.getenv("DB_URL", "sqlite:///data/db/heal.db")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frame_skip: int = int(os.getenv("FRAME_SKIP", "2"))
    min_detection_score: float = float(os.getenv("MIN_DETECTION_SCORE", "0.6"))
    min_face_size: int = int(os.getenv("MIN_FACE_SIZE", "60"))
    attendance_log_interval: int = int(os.getenv("ATTENDANCE_LOG_INTERVAL_SECONDS", "5"))

    def __init__(self):
        # Resolve paths relative to project root
        root = Path(__file__).parent.parent
        self.snapshot_dir = str(root / self.snapshot_dir)
        self.enrolled_photos_dir = str(root / self.enrolled_photos_dir)
        self.faiss_index_path = str(root / self.faiss_index_path)
        self.faiss_map_path = str(root / self.faiss_map_path)

        # Ensure directories exist
        for d in [self.snapshot_dir, self.enrolled_photos_dir,
                  str(root / "data/faiss"), str(root / "data/db")]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
