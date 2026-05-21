"""
Bulk enroll employees from a directory of photos.
Directory structure:
  photos/
    John Doe/
      photo1.jpg
      photo2.jpg
    Alice Smith/
      photo1.jpg

Run: python scripts/enroll_bulk.py --dir /path/to/photos --department Engineering
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import init_db, SessionLocal
from core.models import Employee
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store


def bulk_enroll(photos_dir: str, department: str = None):
    init_db()
    engine = get_face_engine()
    store = get_faiss_store()
    db = SessionLocal()

    photos_path = Path(photos_dir)
    if not photos_path.exists():
        print(f"Directory not found: {photos_dir}")
        sys.exit(1)

    enrolled = 0
    failed = 0

    for person_dir in sorted(photos_path.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        print(f"\nEnrolling: {name}")

        # Check if already enrolled
        existing = db.query(Employee).filter(Employee.name == name).first()
        if existing:
            print(f"  SKIP: {name} already enrolled (ID={existing.id})")
            continue

        # Find a usable photo
        embedding = None
        used_photo = None
        for photo_path in sorted(person_dir.glob("*.jpg")) or sorted(person_dir.glob("*.png")):
            emb = engine.embed_from_path(str(photo_path))
            if emb is not None:
                embedding = emb
                used_photo = photo_path
                break

        if embedding is None:
            print(f"  FAIL: No face detected in any photo for {name}")
            failed += 1
            continue

        # Create employee record
        emp = Employee(
            name=name,
            department=department,
            photo_path=str(used_photo),
        )
        db.add(emp)
        db.commit()
        db.refresh(emp)

        # Add to Faiss
        store.add(emp.id, embedding)
        print(f"  OK: Enrolled as ID={emp.id} using {used_photo.name}")
        enrolled += 1

    db.close()
    print(f"\nDone: {enrolled} enrolled, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk enroll employees from photos directory")
    parser.add_argument("--dir", required=True, help="Path to photos directory")
    parser.add_argument("--department", default=None, help="Department name for all enrollees")
    args = parser.parse_args()
    bulk_enroll(args.dir, args.department)
