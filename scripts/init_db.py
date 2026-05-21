"""
Initialize the database — create all tables.
Run from project root: python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import init_db, engine
from core.config import settings

if __name__ == "__main__":
    print(f"Initializing database at: {settings.db_url}")
    init_db()
    print("Tables created successfully.")

    # Verify
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables: {tables}")
