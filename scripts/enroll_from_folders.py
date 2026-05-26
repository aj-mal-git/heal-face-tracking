"""
Bulk enrollment from a folder of employee photo directories.

Expected folder structure:
  data/enrolled_photos/
    John_Doe/
      front.jpg
      left.jpg
      right.jpg
      cctv_angle1.jpg
      ...
    Jane_Smith/
      img_001.jpg
      img_002.jpg
      ...

Each subfolder name = employee name.
All images inside = different angle photos of that employee.

Usage:
  py -3.13 scripts/enroll_from_folders.py
  py -3.13 scripts/enroll_from_folders.py --photos-dir path/to/photos
  py -3.13 scripts/enroll_from_folders.py --min-score 0.6 --dry-run

How it works:
  1. For each employee folder  → create/find employee in DB
  2. For each photo in folder  → run SCRFD detector + ArcFace embedder
  3. Quality filter            → skip low-confidence or tiny faces
  4. Store ALL embeddings      → FAISS (multiple rows per employee)
  5. Report                    → summary table of results
"""
import sys
import os
import argparse
from pathlib import Path

# Make sure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import settings
from core.database import init_db, get_db
from core.face_engine import get_face_engine
from core.faiss_store import get_faiss_store
from core.models import Employee

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Quality thresholds
MIN_DETECTION_SCORE = 0.60   # reject low-confidence face detections
MIN_FACE_SIZE_PX    = 60     # reject faces smaller than 60×60 pixels


def enroll_from_folders(
    photos_dir: str,
    min_score: float = MIN_DETECTION_SCORE,
    dry_run: bool = False,
):
    photos_root = Path(photos_dir)
    if not photos_root.exists():
        print(f"[ERROR] Directory not found: {photos_root}")
        return

    # Init all components
    print("[Enroll] Initializing database...")
    init_db()
    engine  = get_face_engine()
    store   = get_faiss_store()
    db_gen  = get_db()
    db      = next(db_gen)

    employee_dirs = sorted([d for d in photos_root.iterdir() if d.is_dir()])
    if not employee_dirs:
        print(f"[ERROR] No subfolders found in {photos_root}")
        return

    print(f"\n[Enroll] Found {len(employee_dirs)} employee folders in {photos_root}")
    print(f"[Enroll] Min face detection score: {min_score}")
    print(f"[Enroll] Dry run: {dry_run}\n")
    print(f"{'─'*70}")

    total_enrolled = 0
    total_embeddings = 0

    for emp_dir in employee_dirs:
        emp_name = emp_dir.name.replace("_", " ").replace("-", " ").strip()
        photo_files = [
            f for f in emp_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]

        if not photo_files:
            print(f"  SKIP  {emp_name:<30} — no image files found")
            continue

        # ── Check if employee already exists in DB ─────────────────────────
        existing = db.query(Employee).filter(Employee.name == emp_name).first()
        if existing:
            emp = existing
            action = "UPDATE"
        else:
            emp = None
            action = "NEW"

        # ── Extract embeddings from all photos ─────────────────────────────
        embeddings = []
        skipped = 0

        for photo_path in sorted(photo_files):
            try:
                import cv2
                import numpy as np
                img = cv2.imread(str(photo_path))
                if img is None:
                    skipped += 1
                    continue

                # Run face detection + embedding
                from core.face_engine import FaceResult
                faces = engine._app.get(img)

                if not faces:
                    skipped += 1
                    continue

                # Pick best face (highest detection score)
                best = max(faces, key=lambda f: f.det_score)

                # Quality filter
                if best.det_score < min_score:
                    skipped += 1
                    continue

                x1, y1, x2, y2 = best.bbox.astype(int)
                if (x2 - x1) < MIN_FACE_SIZE_PX or (y2 - y1) < MIN_FACE_SIZE_PX:
                    skipped += 1
                    continue

                embeddings.append(best.normed_embedding)

            except Exception as e:
                print(f"    [WARN] Failed to process {photo_path.name}: {e}")
                skipped += 1

        if not embeddings:
            print(f"  SKIP  {emp_name:<30} — 0 usable faces from {len(photo_files)} photos")
            continue

        if dry_run:
            print(
                f"  DRY   {emp_name:<30} "
                f"{len(embeddings):>3} embeddings  "
                f"({skipped} skipped)  [{action}]"
            )
            continue

        # ── Write to DB ────────────────────────────────────────────────────
        if emp is None:
            emp = Employee(
                name=emp_name,
                photo_path=str(sorted(photo_files)[0]),
            )
            db.add(emp)
            db.commit()
            db.refresh(emp)
        else:
            # Remove old embeddings before re-adding from fresh photo set
            store.remove_employee(emp.id)

        # ── Add all embeddings to FAISS ────────────────────────────────────
        added = store.add_many(emp.id, embeddings)

        print(
            f"  OK    {emp_name:<30} "
            f"{added:>3} embeddings  "
            f"({skipped} photos skipped)  [ID={emp.id}] [{action}]"
        )

        total_enrolled += 1
        total_embeddings += added

    print(f"{'─'*70}")
    if dry_run:
        print(f"[Dry run complete — nothing written to DB or FAISS]")
    else:
        print(f"[Done] {total_enrolled} employees enrolled, {total_embeddings} total embeddings in FAISS.")
        print(f"[Done] FAISS index now has {store.count()} total vectors.")

    try:
        next(db_gen)
    except StopIteration:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk enroll employees from photo folders")
    parser.add_argument(
        "--photos-dir",
        default=str(ROOT / "data" / "enrolled_photos"),
        help="Root folder containing one subfolder per employee",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_DETECTION_SCORE,
        help="Minimum face detection confidence to accept a photo (0.0–1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be enrolled without writing anything",
    )
    args = parser.parse_args()
    enroll_from_folders(args.photos_dir, args.min_score, args.dry_run)
