# backend/app/main.py
import os
import time
import uuid
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import engine, get_db, Base
from .models import User, Document, Verification
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

# ── Logging ────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Create tables ──────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="CertiVerify API", version="1.0.0")

# CORS — allow the frontend (any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Uploads directory ──────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═══════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyRequest(BaseModel):
    document_id: int


class UserResponse(BaseModel):
    id: int
    username: str
    email: str


class AuthResponse(BaseModel):
    access_token: str
    user: UserResponse


# ═══════════════════════════════════════════════════════════════
#  Health Check
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════
#  Auth Routes
# ═══════════════════════════════════════════════════════════════

@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # Check duplicates
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.id})
    return {
        "access_token": token,
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(data={"sub": user.id})
    return {
        "access_token": token,
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


# ═══════════════════════════════════════════════════════════════
#  Document Upload
# ═══════════════════════════════════════════════════════════════

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are allowed")

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB")

    # Save to disk with unique name
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)

    with open(filepath, "wb") as f:
        f.write(content)

    # Create DB record
    document = Document(
        filename=file.filename,
        filepath=filepath,
        file_size=file_size,
        user_id=current_user.id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return {
        "id": document.id,
        "filename": document.filename,
        "file_size": document.file_size,
        "upload_time": document.upload_time.isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
#  Verification (ML-powered Analysis)
# ═══════════════════════════════════════════════════════════════

# Try to load the trained ML model at startup
try:
    from .ml.model import predict as ml_predict
    logger.info("ML model module loaded successfully")
    ML_AVAILABLE = True
except Exception as e:
    logger.warning(f"ML model not available, using fallback: {e}")
    ML_AVAILABLE = False


@app.post("/api/verify")
def verify_document(
    req: VerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Ensure document exists and belongs to user
    document = db.query(Document).filter(
        Document.id == req.document_id,
        Document.user_id == current_user.id,
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Run ML-powered analysis
    start_time = time.time()

    if ML_AVAILABLE:
        try:
            result = ml_predict(document.filepath)
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    else:
        raise HTTPException(
            status_code=503,
            detail="ML model not trained yet. Run 'python -m app.ml.train' from the backend directory."
        )

    # Persist verification
    verification = Verification(
        document_id=document.id,
        user_id=current_user.id,
        is_fake=result["is_fake"],
        confidence=result["confidence"],
        final_score=result["final_score"],
        texture_score=result["texture_score"],
        gan_score=result["gan_score"],
        processing_time_ms=result["processing_time_ms"],
        lbp_entropy=result["texture_features"]["lbp_entropy"],
        glm_contrast=result["texture_features"]["glm_contrast"],
        gabor_energy=result["texture_features"]["gabor_energy"],
    )
    db.add(verification)
    db.commit()
    db.refresh(verification)

    return result


# ═══════════════════════════════════════════════════════════════
#  Verification History  (must be defined before /{verification_id})
# ═══════════════════════════════════════════════════════════════

@app.get("/api/verify/history")
def get_verification_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verifications = (
        db.query(Verification)
        .filter(Verification.user_id == current_user.id)
        .order_by(Verification.verification_time.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []
    for v in verifications:
        doc = db.query(Document).filter(Document.id == v.document_id).first()
        results.append({
            "id": v.id,
            "filename": doc.filename if doc else "Unknown",
            "is_fake": v.is_fake,
            "confidence": v.confidence,
            "final_score": v.final_score,
            "texture_score": v.texture_score,
            "gan_score": v.gan_score,
            "processing_time_ms": v.processing_time_ms,
            "verification_time": v.verification_time.isoformat(),
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  Single Verification Detail
# ═══════════════════════════════════════════════════════════════

@app.get("/api/verify/{verification_id}")
def get_verification_detail(
    verification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verification = db.query(Verification).filter(
        Verification.id == verification_id,
        Verification.user_id == current_user.id,
    ).first()

    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")

    document = db.query(Document).filter(Document.id == verification.document_id).first()

    return {
        "document": {
            "id": document.id if document else None,
            "filename": document.filename if document else "Unknown",
        },
        "verification": {
            "id": verification.id,
            "is_fake": verification.is_fake,
            "confidence": verification.confidence,
            "final_score": verification.final_score,
            "texture_score": verification.texture_score,
            "gan_score": verification.gan_score,
            "processing_time_ms": verification.processing_time_ms,
            "verification_time": verification.verification_time.isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════
#  System Stats
# ═══════════════════════════════════════════════════════════════

@app.get("/api/stats/system")
def get_system_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total = db.query(func.count(Verification.id)).filter(
        Verification.user_id == current_user.id
    ).scalar() or 0

    if total == 0:
        return {
            "total_verifications": 0,
            "fake_detection_rate": 0.0,
            "average_processing_time_ms": 0.0,
            "average_confidence": 0.0,
        }

    fake_count = db.query(func.count(Verification.id)).filter(
        Verification.user_id == current_user.id,
        Verification.is_fake == True,
    ).scalar() or 0

    avg_time = db.query(func.avg(Verification.processing_time_ms)).filter(
        Verification.user_id == current_user.id
    ).scalar() or 0.0

    avg_confidence = db.query(func.avg(Verification.confidence)).filter(
        Verification.user_id == current_user.id
    ).scalar() or 0.0

    return {
        "total_verifications": total,
        "fake_detection_rate": round(fake_count / total, 4),
        "average_processing_time_ms": round(float(avg_time), 2),
        "average_confidence": round(float(avg_confidence), 4),
    }


# ═══════════════════════════════════════════════════════════════
#  Serve Frontend  (must be LAST — after all API routes)
# ═══════════════════════════════════════════════════════════════

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{filename}.html")
def serve_html(filename: str):
    filepath = os.path.join(FRONTEND_DIR, f"{filename}.html")
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Page not found")

# Mount static assets (CSS, JS, images)
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
if os.path.exists(os.path.join(FRONTEND_DIR, "images")):
    app.mount("/images", StaticFiles(directory=os.path.join(FRONTEND_DIR, "images")), name="images")
