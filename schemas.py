# schemas.py — Pydantic v2 request/response validation models
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class AppModel(BaseModel):
    """
    Base for all schemas.
      - protected_namespaces=() silences the model_ field name warning.
      - from_attributes=True enables ORM-mode serialisation everywhere.
    """
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)


# ── Clinical feature schema ────────────────────────────────────────────────────
class ReadingFeatures(AppModel):
    """7 patient-reported clinical symptoms."""
    air_sensation:      float = Field(..., ge=1,  le=10,  description="Perceived air quality (1-10)")
    nasal_dryness:      float = Field(..., ge=1,  le=10,  description="Nasal dryness severity (1-10)")
    nasal_burning:      float = Field(..., ge=1,  le=10,  description="Nasal burning sensation (1-10)")
    suffocation:        float = Field(..., ge=1,  le=10,  description="Suffocation / airflow difficulty (1-10)")
    anxiety_score:      float = Field(..., ge=1,  le=10,  description="Anxiety level (1-10)")
    humidity_level_pct: float = Field(..., ge=0,  le=100, description="Indoor humidity %")
    sleep_quality_hrs:  float = Field(..., ge=0,  le=12,  description="Hours of sleep")


# ── Patient ────────────────────────────────────────────────────────────────────
class PatientCreate(AppModel):
    patient_id:    str
    name:          Optional[str] = None
    first_reading: ReadingFeatures


class PatientResponse(AppModel):
    patient_id:      str
    name:            Optional[str]
    total_readings:  int
    baseline_values: Dict[str, Any]
    created_at:      datetime


# ── Daily Reading ──────────────────────────────────────────────────────────────
class LogReadingRequest(AppModel):
    features:      ReadingFeatures
    signal_actual: Optional[str] = Field(None, pattern="^(Green|Yellow|Red)$")


class LogReadingResponse(AppModel):
    """Returned immediately after a POST /readings call."""
    reading_id:       int
    patient_id:       str
    deviation_values: Dict[str, float]
    signal_actual:    Optional[str]
    logged_at:        datetime
    baseline_updated: bool


class ReadingHistoryResponse(AppModel):
    """Returned by GET /readings — maps directly to DB columns."""
    id:                int
    patient_id:        str
    logged_at:         datetime
    air_sensation:     float
    nasal_dryness:     float
    nasal_burning:     float
    suffocation:       float
    anxiety_score:     float
    humidity_level_pct: float
    sleep_quality_hrs: float
    deviation_values:  Optional[Dict[str, float]]
    signal_actual:     Optional[str]


# ── Prediction ─────────────────────────────────────────────────────────────────
class PredictRequest(AppModel):
    features: ReadingFeatures


class TopDeviation(AppModel):
    feature: str
    z:       float


class PredictResponse(AppModel):
    patient_id:       str
    signal_predicted: str
    ml_signal:        str
    z_override:       bool
    max_abs_z:        float
    z_scores:         Dict[str, float]
    top_deviations:   List[TopDeviation]
    confidence:       float
    prob_green:       float
    prob_yellow:      float
    prob_red:         float
    model_used:       str
    borderline:       bool


class PredictionHistoryResponse(AppModel):
    id:               int
    patient_id:       str
    reading_id:       Optional[int]
    predicted_at:     datetime
    signal_predicted: str
    prob_green:       float
    prob_yellow:      float
    prob_red:         float
    model_used:       str
    confidence:       float


class LogAndPredictResponse(AppModel):
    patient_id:       str
    reading_id:       int
    baseline_updated: bool
    signal_predicted: str
    ml_signal:        str
    z_override:       bool
    max_abs_z:        float
    z_scores:         Dict[str, float]
    top_deviations:   List[TopDeviation]
    confidence:       float
    prob_green:       float
    prob_yellow:      float
    prob_red:         float
    model_used:       str
    borderline:       bool


# ── Agent Intake ───────────────────────────────────────────────────────────────
class AgentIntakeRequest(AppModel):
    """
    Single flat JSON from the agentic AI pipeline.
    All 7 clinical features + patient identity.
    humidity_level_pct accepts 0-100 (not clamped to 1+ like symptom scores).
    signal_actual is optional ground-truth label for supervised learning.
    """
    patient_id:         str
    name:               Optional[str] = None
    air_sensation:      float = Field(..., ge=1,  le=10,  description="Perceived air quality (1-10)")
    nasal_dryness:      float = Field(..., ge=1,  le=10,  description="Nasal dryness severity (1-10)")
    nasal_burning:      float = Field(..., ge=1,  le=10,  description="Nasal burning sensation (1-10)")
    suffocation:        float = Field(..., ge=1,  le=10,  description="Suffocation / airflow difficulty (1-10)")
    anxiety_score:      float = Field(..., ge=1,  le=10,  description="Anxiety level (1-10)")
    humidity_level_pct: float = Field(..., ge=0,  le=100, description="Indoor humidity %")
    sleep_quality_hrs:  float = Field(..., ge=0,  le=12,  description="Hours of sleep")
    signal_actual:      Optional[str] = Field(None, pattern="^(Green|Yellow|Red)$")

    def to_reading_features(self) -> ReadingFeatures:
        return ReadingFeatures(
            air_sensation=      self.air_sensation,
            nasal_dryness=      self.nasal_dryness,
            nasal_burning=      self.nasal_burning,
            suffocation=        self.suffocation,
            anxiety_score=      self.anxiety_score,
            humidity_level_pct= self.humidity_level_pct,
            sleep_quality_hrs=  self.sleep_quality_hrs,
        )


class AgentIntakeResponse(AppModel):
    """Complete response returned to the agentic AI pipeline."""
    patient_id:       str
    reading_id:       int
    is_new_patient:   bool
    signal_predicted: str
    ml_signal:        str
    z_override:       bool
    max_abs_z:        float
    z_scores:         Dict[str, float]
    top_deviations:   List[TopDeviation]
    prob_green:       float
    prob_yellow:      float
    prob_red:         float
    confidence:       float
    model_used:       str
    borderline:       bool
    baseline_updated: bool
    signal_actual:    Optional[str]
    z_summary:        str


# ── Model Metadata ─────────────────────────────────────────────────────────────
class ModelMetaResponse(AppModel):
    patient_id:  str
    model_type:  str
    rows_used:   int
    model_path:  Optional[str]
    trained_at:  datetime
    cv_f1_score: Optional[float]
    notes:       Optional[str]


# ── Stats ──────────────────────────────────────────────────────────────────────
class PatientStatsResponse(AppModel):
    patient_id:          str
    total_readings:      int
    signal_breakdown:    Dict[str, int]
    avg_sleep:           float
    avg_anxiety:         float
    model_type_active:   str
    rows_until_personal: Optional[int]
