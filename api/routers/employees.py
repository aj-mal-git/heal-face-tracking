"""
Employee enrollment and management endpoints.
POST /api/employees/enroll  — register a new employee with photo
GET  /api/employees/        — list all employees
GET  /api/employees/{id}    — get single employee
PUT  /api/employees/{id}    — update employee info
DELETE /api/employees/{id}  — deactivate employee
POST /api/employees/{id}/re-enroll — update face embedding
"""
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import Employee
from core.schemas import EmployeeRead, EmployeeUpdate
from core.face_engine import FaceEngine, get_face_engine
from core.faiss_store import FaissStore, get_faiss_store
from core.config import settings

router = APIRouter()


@router.post("/enroll", response_model=EmployeeRead)
async def enroll_employee(
    name: str = Form(...),
    department: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    photos: List[UploadFile] = File(...),   # accepts 1 or many photos
    db: Session = Depends(get_db),
    engine: FaceEngine = Depends(get_face_engine),
    store: FaissStore = Depends(get_faiss_store),
):
    """
    Register a new employee with one or more photos (all angles welcome).
    Each photo that contains a detectable face adds one embedding to FAISS.
    More angle photos = better recognition accuracy from any CCTV angle.
    """
    emp_dir = os.path.join(settings.enrolled_photos_dir, f"emp_{uuid.uuid4().hex[:8]}")
    os.makedirs(emp_dir, exist_ok=True)

    embeddings = []
    first_photo_path = None

    for photo in photos:
        photo_bytes = await photo.read()
        embedding = engine.embed_from_bytes(photo_bytes)
        if embedding is None:
            continue  # skip photos with no detectable face

        # Save photo to disk
        photo_path = os.path.join(emp_dir, f"{uuid.uuid4().hex}.jpg")
        with open(photo_path, "wb") as f:
            f.write(photo_bytes)
        if first_photo_path is None:
            first_photo_path = photo_path
        embeddings.append(embedding)

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="No face detected in any of the uploaded photos. Use clear, well-lit images.",
        )

    # Write employee to DB
    employee = Employee(
        name=name,
        department=department,
        email=email,
        photo_path=first_photo_path,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)

    # Add ALL embeddings to FAISS (one per angle photo)
    added = store.add_many(employee.id, embeddings)
    print(f"[Enrollment] Enrolled '{name}' (ID={employee.id}) with {added} embeddings from {len(photos)} photos.")
    return employee


@router.get("/", response_model=List[EmployeeRead])
def list_employees(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    db: Session = Depends(get_db),
):
    query = db.query(Employee)
    if active_only:
        query = query.filter(Employee.is_active == True)
    return query.offset(skip).limit(limit).all()


@router.get("/{employee_id}", response_model=EmployeeRead)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp


@router.put("/{employee_id}", response_model=EmployeeRead)
def update_employee(
    employee_id: int,
    update: EmployeeUpdate,
    db: Session = Depends(get_db),
):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(emp, field, value)
    db.commit()
    db.refresh(emp)
    return emp


@router.delete("/{employee_id}")
def deactivate_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    store: FaissStore = Depends(get_faiss_store),
):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    emp.is_active = False
    db.commit()
    store.remove_employee(employee_id)
    return {"message": f"Employee {employee_id} deactivated and removed from recognition index."}


@router.post("/{employee_id}/re-enroll", response_model=EmployeeRead)
async def re_enroll(
    employee_id: int,
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: FaceEngine = Depends(get_face_engine),
    store: FaissStore = Depends(get_faiss_store),
):
    """Replace an employee's face embedding with a new photo."""
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    photo_bytes = await photo.read()
    embedding = engine.embed_from_bytes(photo_bytes)
    if embedding is None:
        raise HTTPException(status_code=422, detail="No face detected in photo.")

    # Remove old embeddings and add new one
    store.remove_employee(employee_id)
    store.add(employee_id, embedding)

    # Save new photo
    emp_dir = os.path.join(settings.enrolled_photos_dir, f"emp_{employee_id}")
    os.makedirs(emp_dir, exist_ok=True)
    photo_path = os.path.join(emp_dir, f"{uuid.uuid4().hex}.jpg")
    with open(photo_path, "wb") as f:
        f.write(photo_bytes)
    emp.photo_path = photo_path
    db.commit()
    db.refresh(emp)
    return emp


@router.post("/{employee_id}/add-photos")
async def add_photos(
    employee_id: int,
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    engine: FaceEngine = Depends(get_face_engine),
    store: FaissStore = Depends(get_faiss_store),
):
    """
    Add more angle photos to an existing employee's gallery WITHOUT removing
    their current embeddings. Use this to keep adding CCTV angle shots over time.
    """
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    embeddings = []
    for photo in photos:
        photo_bytes = await photo.read()
        emb = engine.embed_from_bytes(photo_bytes)
        if emb is None:
            continue
        emp_dir = os.path.join(settings.enrolled_photos_dir, f"emp_{employee_id}")
        os.makedirs(emp_dir, exist_ok=True)
        photo_path = os.path.join(emp_dir, f"{uuid.uuid4().hex}.jpg")
        with open(photo_path, "wb") as f:
            f.write(photo_bytes)
        embeddings.append(emb)

    if not embeddings:
        raise HTTPException(status_code=422, detail="No faces detected in any uploaded photo.")

    added = store.add_many(employee_id, embeddings)
    total = store.count_for_employee(employee_id)
    return {
        "employee_id": employee_id,
        "name": emp.name,
        "embeddings_added": added,
        "total_embeddings": total,
    }


@router.get("/{employee_id}/embedding-count")
def embedding_count(
    employee_id: int,
    db: Session = Depends(get_db),
    store: FaissStore = Depends(get_faiss_store),
):
    """How many angle-embeddings are stored for this employee."""
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {
        "employee_id": employee_id,
        "name": emp.name,
        "total_embeddings": store.count_for_employee(employee_id),
    }
