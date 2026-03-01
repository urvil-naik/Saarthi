# ml_service.py — ML logic: Welford Z-score baseline, prediction, personal model training
#
# Features: 7 patient-reported clinical symptoms only.
#   Air_Sensation, Nasal_Dryness, Nasal_Burning, Suffocation,
#   Anxiety_Score, Humidity_Level_Pct, Sleep_Quality_Hrs
from __future__ import annotations

import math
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from config import settings

logger = logging.getLogger(__name__)

# ── Feature registry ───────────────────────────────────────────────────────────
FEATURES = [
    "Air_Sensation",
    "Nasal_Dryness",
    "Nasal_Burning",
    "Suffocation",
    "Anxiety_Score",
    "Humidity_Level_Pct",
    "Sleep_Quality_Hrs",
]

# Maps lowercase DB column names → canonical Title_Snake feature names
DB_TO_FEATURE = {
    "air_sensation":      "Air_Sensation",
    "nasal_dryness":      "Nasal_Dryness",
    "nasal_burning":      "Nasal_Burning",
    "suffocation":        "Suffocation",
    "anxiety_score":      "Anxiety_Score",
    "humidity_level_pct": "Humidity_Level_Pct",
    "sleep_quality_hrs":  "Sleep_Quality_Hrs",
}

# Neutral fallback defaults when a feature value is missing
FEATURE_DEFAULTS = {
    "air_sensation":      5.0,
    "nasal_dryness":      5.0,
    "nasal_burning":      5.0,
    "suffocation":        5.0,
    "anxiety_score":      5.0,
    "humidity_level_pct": 50.0,
    "sleep_quality_hrs":  7.0,
}

# Z-score signal thresholds
Z_RED_THRESHOLD    = 3.0
Z_YELLOW_THRESHOLD = 2.0

MODEL_DIR          = Path(settings.MODEL_DIR)
MODEL_DIR.mkdir(exist_ok=True)
GLOBAL_MODEL_PATH  = MODEL_DIR / "global_model.pkl"
SCALER_PATH        = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ── Module-level artifact cache ────────────────────────────────────────────────
# Artifacts are loaded lazily on first use and then reused for the lifetime of
# the process — avoiding a disk read on every prediction call.
_global_model   = None
_global_scaler  = None
_label_encoder  = None


def _get_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}.")
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _label_encoder


def _get_global_model():
    global _global_model
    if _global_model is None:
        if not GLOBAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Global model not found at {GLOBAL_MODEL_PATH}.")
        _global_model = joblib.load(GLOBAL_MODEL_PATH)
    return _global_model


def _get_global_scaler():
    global _global_scaler
    if _global_scaler is None and SCALER_PATH.exists():
        _global_scaler = joblib.load(SCALER_PATH)
    return _global_scaler


# ── Public loaders (kept for backward-compat / training scripts) ───────────────
def load_global_model():
    return _get_global_model()


def load_scaler():
    return _get_global_scaler()


def load_label_encoder():
    return _get_label_encoder()


# ── Welford Online Algorithm ───────────────────────────────────────────────────
def _welford_init(value: float) -> dict:
    return {"mean": round(float(value), 4), "m2": 0.0, "n": 1}


def _coerce_state(state, default_col: str) -> dict:
    """
    Guarantee state is a valid Welford dict {mean, m2, n}.

    Handles three legacy / corrupt cases:
      • None / missing key          → fresh init from FEATURE_DEFAULTS
      • float (old flat-value fmt)  → treat the float as the mean, n=1, m2=0
      • dict missing required keys  → patch in safe defaults

    This makes all Welford consumers tolerant of stale DB rows produced
    before the current schema was in place, so a DB wipe is never required
    after a code update.

    NOTE: The `or` short-circuit is intentionally NOT used here because a
    state whose mean == 0.0 is falsy but perfectly valid; `or` would silently
    discard real data in that edge case.
    """
    if state is None:
        return _welford_init(FEATURE_DEFAULTS[default_col])
    if isinstance(state, (int, float)):
        # Legacy format: the value itself was stored directly as the mean
        return {"mean": round(float(state), 4), "m2": 0.0, "n": 1}
    if isinstance(state, dict):
        # Patch any missing keys defensively
        return {
            "mean": float(state.get("mean", FEATURE_DEFAULTS[default_col])),
            "m2":   float(state.get("m2",   0.0)),
            "n":    int(  state.get("n",     1)),
        }
    # Unexpected type — fall back to neutral default
    logger.warning("Unexpected Welford state type %s for col %s; reinitialising.",
                   type(state).__name__, default_col)
    return _welford_init(FEATURE_DEFAULTS[default_col])


def _welford_update(state: dict, new_value: float) -> dict:
    n     = state["n"] + 1
    mean  = state["mean"]
    m2    = state["m2"]
    x     = float(new_value)
    delta = x - mean
    mean += delta / n
    m2   += delta * (x - mean)
    return {"mean": round(mean, 4), "m2": round(m2, 6), "n": n}


def _welford_std(state: dict) -> float:
    n, m2 = state["n"], state["m2"]
    if n < 2 or m2 <= 0:
        return 0.0
    return math.sqrt(m2 / (n - 1))


def _z_score(value: float, state: dict) -> float:
    std = _welford_std(state)
    return round((float(value) - state["mean"]) / std, 3) if std != 0.0 else 0.0


# ── Baseline & Deviation Logic ─────────────────────────────────────────────────
def initialize_baseline(features: dict) -> dict:
    """
    Seed Welford baseline from the patient's first reading.
    Any missing feature key falls back to FEATURE_DEFAULTS.
    """
    return {
        DB_TO_FEATURE[col]: _welford_init(features.get(col, FEATURE_DEFAULTS[col]))
        for col in DB_TO_FEATURE
    }


def update_baseline(current_baseline: dict, new_features: dict) -> dict:
    """
    Welford update for all 7 features.
    Must be called AFTER compute_deviations() — standard Welford pattern:
    compute Z against current baseline, THEN update with the new observation.
    """
    updated = {}
    for db_col, feat_name in DB_TO_FEATURE.items():
        value = float(new_features.get(db_col, FEATURE_DEFAULTS[db_col]))
        # _coerce_state heals any legacy / corrupt baseline entry before updating
        state = _coerce_state(current_baseline.get(feat_name), db_col)
        updated[feat_name] = _welford_update(state, value)
    return updated


def compute_deviations(features: dict, baseline: dict) -> dict:
    """
    Returns Z-scores for all 7 clinical features.
    Input keys: lowercase DB column names.
    Output keys: Title_Snake feature names.
    """
    z_scores = {}
    for db_col, feat_name in DB_TO_FEATURE.items():
        raw   = float(features.get(db_col, FEATURE_DEFAULTS[db_col]))
        # _coerce_state heals None, float, or malformed dict — never crashes
        state = _coerce_state(baseline.get(feat_name), db_col)
        z_scores[feat_name] = _z_score(raw, state)
    return z_scores


# ── Z-score Rule Engine ────────────────────────────────────────────────────────
def z_score_signal(z_scores: dict) -> str | None:
    """
    Pure clinical rule engine.

    Rules (applied in order):
      1. Any feature Z >= 3.0  → Red
      2. Any feature Z >= 2.0  → Yellow
      3. Otherwise             → None  (ML model output used as-is)
    """
    max_z = max(abs(z_scores.get(k, 0.0)) for k in FEATURES)

    if max_z >= Z_RED_THRESHOLD:
        return "Red"
    if max_z >= Z_YELLOW_THRESHOLD:
        return "Yellow"
    return None


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(patient_id: str, features: dict, baseline: dict | None = None) -> dict:
    """
    Predict ENS signal for a patient.

    features: dict with lowercase DB column name keys.
    baseline: current Welford state dict for the patient.

    Uses personal model if one exists for this patient, otherwise global model.
    Each model is paired with its own scaler saved at training time.

    Global model and label encoder are loaded once per process via module-level
    cache (_get_global_model, _get_label_encoder) to avoid repeated disk reads.
    Personal models are loaded per-call since they are patient-specific and
    updated infrequently.
    """
    personal_path        = MODEL_DIR / f"{patient_id}_model.pkl"
    personal_scaler_path = MODEL_DIR / f"{patient_id}_scaler.pkl"
    le                   = _get_label_encoder()

    if personal_path.exists():
        model      = joblib.load(personal_path)
        model_used = "personal"
        scaler     = joblib.load(personal_scaler_path) if personal_scaler_path.exists() else None
    else:
        model      = _get_global_model()
        model_used = "global"
        scaler     = _get_global_scaler()

    # Build feature row — 7 clinical features only
    row     = {DB_TO_FEATURE[col]: float(features.get(col, FEATURE_DEFAULTS[col])) for col in DB_TO_FEATURE}
    X       = pd.DataFrame([row])[FEATURES]
    X_input = scaler.transform(X) if scaler is not None else X.values

    probs     = model.predict_proba(X_input)[0]
    ml_label  = le.inverse_transform([model.predict(X_input)[0]])[0]
    prob_dict = dict(zip(le.classes_, probs.tolist()))

    # Z-score analysis — compute BEFORE baseline update (called in crud.log_reading)
    z_scores    = compute_deviations(features, baseline or {})
    z_override  = z_score_signal(z_scores)
    final_label = z_override if z_override else ml_label

    top_features = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    return {
        "signal_predicted": final_label,
        "ml_signal":        ml_label,
        "z_override":       z_override is not None,
        "max_abs_z":        round(max(abs(v) for v in z_scores.values()), 3) if z_scores else 0.0,
        "z_scores":         {k: round(v, 3) for k, v in z_scores.items()},
        "top_deviations":   [{"feature": k, "z": round(v, 3)} for k, v in top_features],
        "prob_green":       round(prob_dict.get("Green",  0.0), 4),
        "prob_yellow":      round(prob_dict.get("Yellow", 0.0), 4),
        "prob_red":         round(prob_dict.get("Red",    0.0), 4),
        "confidence":       round(float(probs.max()), 4),
        "model_used":       model_used,
        "borderline":       float(probs.max()) < 0.60 and z_override is None,
    }


# ── Personal Model Training ────────────────────────────────────────────────────
def train_personal_model(patient_id: str, readings: list[dict]):
    """
    Train a personal RF model from the patient's labeled readings.

    Returns (model, cv_f1, rows_used) or (None, None, row_count) if
    insufficient labeled rows exist.

    A personal StandardScaler is fitted on this patient's own data and saved
    alongside the model so inference is always consistent.
    """
    le = _get_label_encoder()
    df = pd.DataFrame(readings).rename(columns=DB_TO_FEATURE)
    df = df[df["Signal"].notna()]

    if len(df) < settings.PERSONAL_MODEL_MIN_ROWS:
        return None, None, len(df)

    # Ensure all 7 feature columns are present; fill missing with defaults
    fill_vals = {DB_TO_FEATURE[col]: FEATURE_DEFAULTS[col] for col in DB_TO_FEATURE}
    for feat_name, default_val in fill_vals.items():
        if feat_name not in df.columns:
            df[feat_name] = default_val

    X = df[FEATURES].fillna(pd.Series(fill_vals))
    y = le.transform(df["Signal"])

    personal_scaler = StandardScaler()
    X_scaled        = personal_scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    try:
        cv        = StratifiedKFold(n_splits=min(5, len(df) // 5), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_weighted")
        cv_f1     = round(float(cv_scores.mean()), 4)
    except Exception as exc:
        logger.warning("Personal model CV failed for %s: %s", patient_id, exc)
        cv_f1 = None

    model.fit(X_scaled, y)

    joblib.dump(model,          MODEL_DIR / f"{patient_id}_model.pkl")
    joblib.dump(personal_scaler, MODEL_DIR / f"{patient_id}_scaler.pkl")

    return model, cv_f1, len(df)
