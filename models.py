# models.py — SQLAlchemy ORM table definitions

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class PatientProfile(Base):
    """
    One row per patient.
    baseline_values: Welford state dict per feature — {feature: {mean, m2, n}}.
    All 7 clinical features are seeded from the patient's first reading.
    """
    __tablename__ = "patient_profiles"

    patient_id      = Column(String, primary_key=True, index=True)
    name            = Column(String, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    baseline_values = Column(JSON, nullable=False)
    total_readings  = Column(Integer, default=0, nullable=False, server_default="0")

    readings    = relationship("DailyReading",  back_populates="patient",  cascade="all, delete-orphan")
    predictions = relationship("Prediction",    back_populates="patient",  cascade="all, delete-orphan")
    models      = relationship("ModelMetadata", back_populates="patient",  cascade="all, delete-orphan")


class DailyReading(Base):
    """
    One row per daily patient report.

    All 7 features are patient-reported clinical indicators:
        air_sensation      — perceived air quality / sensation (1–10)
        nasal_dryness      — nasal dryness severity (1–10)
        nasal_burning      — nasal burning sensation (1–10)
        suffocation        — suffocation / airflow difficulty (1–10)
        anxiety_score      — anxiety level (1–10)
        humidity_level_pct — indoor humidity percentage (0–100)
        sleep_quality_hrs  — hours of sleep (0–12)
    """
    __tablename__ = "daily_readings"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patient_profiles.patient_id", ondelete="CASCADE"), index=True)
    logged_at  = Column(DateTime(timezone=True), server_default=func.now())

    # ── Patient-reported clinical features ────────────────────────────────────
    air_sensation      = Column(Float, nullable=False)
    nasal_dryness      = Column(Float, nullable=False)
    nasal_burning      = Column(Float, nullable=False)
    suffocation        = Column(Float, nullable=False)
    anxiety_score      = Column(Float, nullable=False)
    humidity_level_pct = Column(Float, nullable=False)
    sleep_quality_hrs  = Column(Float, nullable=False)

    # ── Computed ───────────────────────────────────────────────────────────────
    deviation_values = Column(JSON, nullable=True)   # per-feature Z-scores
    signal_actual    = Column(String, nullable=True) # Green / Yellow / Red

    patient     = relationship("PatientProfile", back_populates="readings")
    predictions = relationship("Prediction",     back_populates="reading", cascade="all, delete-orphan")


# Composite index: fast ORDER BY logged_at DESC per patient
Index("ix_daily_readings_patient_logged", DailyReading.patient_id, DailyReading.logged_at)


class Prediction(Base):
    """One row per prediction — ML output + Z-score override metadata."""
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    patient_id       = Column(String,  ForeignKey("patient_profiles.patient_id", ondelete="CASCADE"), index=True)
    reading_id       = Column(Integer, ForeignKey("daily_readings.id", ondelete="SET NULL"), nullable=True)
    predicted_at     = Column(DateTime(timezone=True), server_default=func.now())

    signal_predicted = Column(String, nullable=False)
    prob_green       = Column(Float,  nullable=False)
    prob_yellow      = Column(Float,  nullable=False)
    prob_red         = Column(Float,  nullable=False)
    model_used       = Column(String, nullable=False)
    confidence       = Column(Float,  nullable=False)

    patient = relationship("PatientProfile", back_populates="predictions")
    reading = relationship("DailyReading",   back_populates="predictions")


class ModelMetadata(Base):
    """Training history for personal models."""
    __tablename__ = "model_metadata"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    patient_id  = Column(String, ForeignKey("patient_profiles.patient_id", ondelete="CASCADE"), index=True)
    trained_at  = Column(DateTime(timezone=True), server_default=func.now())
    model_type  = Column(String, nullable=False)
    rows_used   = Column(Integer, nullable=False)
    model_path  = Column(String, nullable=True)
    cv_f1_score = Column(Float,  nullable=True)
    notes       = Column(Text,   nullable=True)

    patient = relationship("PatientProfile", back_populates="models")
