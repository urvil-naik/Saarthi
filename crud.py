# crud.py — all database read/write operations
from __future__ import annotations

from pathlib import Path
from sqlalchemy.orm import Session
import models, schemas, ml_service
from config import settings


# ── Patient ────────────────────────────────────────────────────────────────────
def get_patient(db: Session, patient_id: str):
    return db.query(models.PatientProfile).filter(
        models.PatientProfile.patient_id == patient_id
    ).first()


def create_patient(db: Session, data: schemas.PatientCreate):
    """Register new patient and seed Welford baseline from first reading."""
    features_dict = data.first_reading.model_dump()
    baseline      = ml_service.initialize_baseline(features_dict)

    patient = models.PatientProfile(
        patient_id=     data.patient_id,
        name=           data.name,
        baseline_values=baseline,
        total_readings= 0,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def update_baseline(db: Session, patient: models.PatientProfile, features: dict):
    """
    Welford update for all 7 clinical features.
    Must be called AFTER compute_deviations() — standard Welford pattern.
    """
    new_baseline            = ml_service.update_baseline(patient.baseline_values, features)
    patient.baseline_values = new_baseline
    db.commit()
    db.refresh(patient)
    return new_baseline


# ── Daily Reading ──────────────────────────────────────────────────────────────
def log_reading(
    db: Session,
    patient_id: str,
    data: schemas.LogReadingRequest,
    skip_baseline_update: bool = False,
) -> models.DailyReading:
    """
    Log a daily reading.

    Pipeline:
      1. Extract patient-reported clinical features.
      2. Compute Z-score deviations against the CURRENT Welford baseline.
         (Must be BEFORE baseline update — standard Welford pattern.)
      3. Store the reading with deviation values.
      4. Update Welford baseline with this new observation
         (skipped when skip_baseline_update=True, e.g. for a new patient
          whose baseline was already seeded by create_patient from this
          same reading — updating again would double-count it).

    Raises ValueError if the patient does not exist (callers should
    have already validated and returned 404, but this prevents a silent
    None return that would cause AttributeError downstream).
    """
    patient = get_patient(db, patient_id)
    if not patient:
        raise ValueError(f"Patient '{patient_id}' not found.")

    features = data.features.model_dump()

    # Step 2: compute Z-scores against current baseline BEFORE updating it
    deviations = ml_service.compute_deviations(features, patient.baseline_values)

    # Step 3: store reading
    reading = models.DailyReading(
        patient_id=        patient_id,
        air_sensation=     features["air_sensation"],
        nasal_dryness=     features["nasal_dryness"],
        nasal_burning=     features["nasal_burning"],
        suffocation=       features["suffocation"],
        anxiety_score=     features["anxiety_score"],
        humidity_level_pct=features["humidity_level_pct"],
        sleep_quality_hrs= features["sleep_quality_hrs"],
        deviation_values=  deviations,
        signal_actual=     data.signal_actual,
    )
    db.add(reading)

    # Step 4: update Welford baseline (skipped for new patients to avoid
    # double-counting: create_patient already seeded the baseline from this
    # same first reading, so calling update here would count it twice,
    # jumping Welford n from 1 → 2 on a single observation.)
    if not skip_baseline_update:
        update_baseline(db, patient, features)

    # Atomic SQL-level increment avoids read-modify-write race under concurrency
    db.query(models.PatientProfile).filter(
        models.PatientProfile.patient_id == patient_id
    ).update(
        {models.PatientProfile.total_readings: models.PatientProfile.total_readings + 1},
        synchronize_session="fetch",
    )

    db.commit()
    db.refresh(reading)
    return reading


def get_patient_readings(db: Session, patient_id: str, limit: int = 100):
    return db.query(models.DailyReading).filter(
        models.DailyReading.patient_id == patient_id
    ).order_by(models.DailyReading.logged_at.desc()).limit(limit).all()


def get_labeled_readings(db: Session, patient_id: str):
    """Returns all labeled readings as dicts for personal model training."""
    rows = db.query(models.DailyReading).filter(
        models.DailyReading.patient_id == patient_id,
        models.DailyReading.signal_actual.isnot(None),
    ).all()

    return [{
        "air_sensation":      r.air_sensation,
        "nasal_dryness":      r.nasal_dryness,
        "nasal_burning":      r.nasal_burning,
        "suffocation":        r.suffocation,
        "anxiety_score":      r.anxiety_score,
        "humidity_level_pct": r.humidity_level_pct,
        "sleep_quality_hrs":  r.sleep_quality_hrs,
        "Signal":             r.signal_actual,
    } for r in rows]


def get_all_readings_for_stats(db: Session, patient_id: str):
    """Returns ALL readings for computing sleep/anxiety averages."""
    return db.query(models.DailyReading).filter(
        models.DailyReading.patient_id == patient_id
    ).all()


# ── Prediction ─────────────────────────────────────────────────────────────────
def save_prediction(db: Session, patient_id: str, reading_id: int | None, pred: dict):
    prediction = models.Prediction(
        patient_id=       patient_id,
        reading_id=       reading_id,
        signal_predicted= pred["signal_predicted"],
        prob_green=       pred["prob_green"],
        prob_yellow=      pred["prob_yellow"],
        prob_red=         pred["prob_red"],
        confidence=       pred["confidence"],
        model_used=       pred["model_used"],
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_patient_predictions(db: Session, patient_id: str, limit: int = 50):
    return db.query(models.Prediction).filter(
        models.Prediction.patient_id == patient_id
    ).order_by(models.Prediction.predicted_at.desc()).limit(limit).all()


# ── Model Metadata ─────────────────────────────────────────────────────────────
def save_model_metadata(
    db: Session,
    patient_id: str,
    model_type: str,
    rows_used: int,
    model_path: str,
    cv_f1: float = None,
    notes: str = None,
):
    meta = models.ModelMetadata(
        patient_id=  patient_id,
        model_type=  model_type,
        rows_used=   rows_used,
        model_path=  model_path,
        cv_f1_score= cv_f1,
        notes=       notes,
    )
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return meta


def get_model_history(db: Session, patient_id: str):
    return db.query(models.ModelMetadata).filter(
        models.ModelMetadata.patient_id == patient_id
    ).order_by(models.ModelMetadata.trained_at.desc()).all()


# ── Stats ──────────────────────────────────────────────────────────────────────
def get_patient_stats(db: Session, patient_id: str):
    patient = get_patient(db, patient_id)
    if not patient:
        return None

    labeled_readings = get_labeled_readings(db, patient_id)
    all_readings     = get_all_readings_for_stats(db, patient_id)
    total            = patient.total_readings or 0

    breakdown = {"Green": 0, "Yellow": 0, "Red": 0}
    for r in labeled_readings:
        if r.get("Signal") in breakdown:
            breakdown[r["Signal"]] += 1

    # Averages computed from ALL readings, not just labeled ones
    sleep_vals   = [r.sleep_quality_hrs for r in all_readings if r.sleep_quality_hrs  is not None]
    anxiety_vals = [r.anxiety_score     for r in all_readings if r.anxiety_score      is not None]

    personal_exists = (Path(settings.MODEL_DIR) / f"{patient_id}_model.pkl").exists()
    labeled_count   = len(labeled_readings)

    return {
        "patient_id":          patient_id,
        "total_readings":      total,
        "signal_breakdown":    breakdown,
        "avg_sleep":           round(sum(sleep_vals)   / len(sleep_vals),   2) if sleep_vals   else 0.0,
        "avg_anxiety":         round(sum(anxiety_vals) / len(anxiety_vals), 2) if anxiety_vals else 0.0,
        "model_type_active":   "personal" if personal_exists else "global",
        "rows_until_personal": None if personal_exists else max(0, settings.PERSONAL_MODEL_MIN_ROWS - labeled_count),
    }
