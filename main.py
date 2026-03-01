# main.py — FastAPI app — ENS prediction API (agent-intake only)
from __future__ import annotations

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path

import crud, schemas, ml_service
from database import get_db, create_tables
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level path refs
_GLOBAL_MODEL_PATH = Path(settings.MODEL_DIR) / "global_model.pkl"
_SCALER_PATH       = Path(settings.MODEL_DIR) / "scaler.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info("Tables verified / created.")
    # Eagerly validate models exist so startup fails fast rather than at first request
    if not _GLOBAL_MODEL_PATH.exists():
        logger.warning("global_model.pkl not found — run train_global_model.py first!")
    if not _SCALER_PATH.exists():
        logger.warning("scaler.pkl not found — run train_global_model.py first!")
    yield


app = FastAPI(
    title="Saarthi — ENS Prediction API",
    description=(
        "Personalized ENS signal prediction based on 7 patient-reported clinical symptoms. "
        "Single endpoint: POST /agent-intake"
    ),
    version="3.1.0",
    lifespan=lifespan,
)

# Allow Flask frontend (port 5000) to call FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Private helpers ────────────────────────────────────────────────────────────

def _maybe_train_personal(patient_id: str, db: Session) -> None:
    """Auto-train personal model when the labeled-reading threshold is met."""
    labeled = crud.get_labeled_readings(db, patient_id)
    if len(labeled) < settings.PERSONAL_MODEL_MIN_ROWS:
        return
    try:
        model, cv_f1, rows_used = ml_service.train_personal_model(patient_id, labeled)
        if model is not None:
            model_path = str(Path(settings.MODEL_DIR) / f"{patient_id}_model.pkl")
            crud.save_model_metadata(
                db, patient_id,
                model_type="personal",
                rows_used=rows_used,
                model_path=model_path,
                cv_f1=cv_f1,
                notes=f"Auto-trained after {len(labeled)} labeled readings.",
            )
            logger.info("Personal model trained for %s (rows=%d, cv_f1=%.4f)",
                        patient_id, rows_used, cv_f1 or 0)
    except Exception as exc:
        logger.warning("Personal model training failed for %s: %s", patient_id, exc)


def _z_summary(z_scores: dict, top_devs: list, max_abs_z: float) -> str:
    if max_abs_z < 2.0:
        return "All clinical features are within your normal range."
    severity = "critically" if max_abs_z >= 3.0 else "notably"
    names    = ", ".join(d["feature"] for d in top_devs if abs(d["z"]) >= 2.0)
    return f"{names} {severity} deviated from your personal baseline (max |Z| = {max_abs_z})."


# ══════════════════════════════════════════════════════════════════════════════
# AGENT INTAKE  —  the ONE endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/agent-intake", response_model=schemas.AgentIntakeResponse, tags=["Agent"])
def agent_intake(data: schemas.AgentIntakeRequest, db: Session = Depends(get_db)):
    """
    Primary (and only) endpoint for the agentic AI pipeline.

    Flow:
      1. Auto-register patient if new (seeds Welford baseline from first reading).
      2. Log daily reading with Z-score deviations computed against current baseline.
      3. Optionally train personal model if threshold reached.
      4. Predict ENS signal (personal model if exists, else global).
      5. Return full result including Z-scores, probabilities, and human-readable summary.

    skip_baseline_update=True for new patients to avoid double-counting:
    create_patient() already seeds baseline from this first reading.
    """
    patient = crud.get_patient(db, data.patient_id)
    is_new  = patient is None

    if is_new:
        patient = crud.create_patient(db, schemas.PatientCreate(
            patient_id=    data.patient_id,
            name=          data.name,
            first_reading= data.to_reading_features(),
        ))
        logger.info("New patient registered: %s (%s)", data.patient_id, data.name)

    reading = crud.log_reading(
        db, data.patient_id,
        schemas.LogReadingRequest(
            features=      data.to_reading_features(),
            signal_actual= data.signal_actual,
        ),
        skip_baseline_update=is_new,
    )

    _maybe_train_personal(data.patient_id, db)

    # Reload patient to get Welford baseline as updated by log_reading
    patient = crud.get_patient(db, data.patient_id)

    try:
        pred = ml_service.predict(
            data.patient_id,
            data.to_reading_features().model_dump(),
            patient.baseline_values,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    crud.save_prediction(db, data.patient_id, reading_id=reading.id, pred=pred)

    logger.info(
        "agent-intake: patient=%s signal=%s z_override=%s confidence=%.2f model=%s",
        data.patient_id, pred["signal_predicted"],
        pred["z_override"], pred["confidence"], pred["model_used"],
    )

    return {
        "patient_id":       data.patient_id,
        "reading_id":       reading.id,
        "is_new_patient":   is_new,
        "signal_predicted": pred["signal_predicted"],
        "ml_signal":        pred["ml_signal"],
        "z_override":       pred["z_override"],
        "max_abs_z":        pred["max_abs_z"],
        "z_scores":         pred["z_scores"],
        "top_deviations":   pred["top_deviations"],
        "prob_green":       pred["prob_green"],
        "prob_yellow":      pred["prob_yellow"],
        "prob_red":         pred["prob_red"],
        "confidence":       pred["confidence"],
        "model_used":       pred["model_used"],
        "borderline":       pred["borderline"],
        "baseline_updated": True,
        "signal_actual":    data.signal_actual,
        "z_summary":        _z_summary(pred["z_scores"], pred["top_deviations"], pred["max_abs_z"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """Returns system health including model and DB status."""
    from sqlalchemy import text
    db_ok = False
    try:
        with next(get_db()) as _:
            pass
        db_ok = True
    except Exception:
        pass

    return {
        "status":                   "ok",
        "version":                  "3.1.0",
        "global_model_loaded":      _GLOBAL_MODEL_PATH.exists(),
        "scaler_loaded":            _SCALER_PATH.exists(),
        "model_dir":                settings.MODEL_DIR,
        "feature_count":            len(ml_service.FEATURES),
        "features":                 ml_service.FEATURES,
        "personal_model_threshold": settings.PERSONAL_MODEL_MIN_ROWS,
        "z_red_threshold":          ml_service.Z_RED_THRESHOLD,
        "z_yellow_threshold":       ml_service.Z_YELLOW_THRESHOLD,
    }
