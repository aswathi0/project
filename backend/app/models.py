# backend/app/models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    documents = relationship("Document", back_populates="owner")
    verifications = relationship("Verification", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", back_populates="documents")
    verifications = relationship("Verification", back_populates="document")


class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Verification results
    is_fake = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    texture_score = Column(Float, nullable=True)
    gan_score = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=False)

    # Texture feature details
    lbp_entropy = Column(Float, nullable=True)
    glm_contrast = Column(Float, nullable=True)
    gabor_energy = Column(Float, nullable=True)

    verification_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="verifications")
    user = relationship("User", back_populates="verifications")
