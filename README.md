# Saarthi / Aria — ENS Health Monitoring System

Two services working together:
- **FastAPI backend** (`main.py`) — single `/agent-intake` endpoint, PostgreSQL + ML
- **Flask frontend** (`app.py`) — Retell AI calling agent + web UI

---

## Quick Start

### 1. Environment
```
cp env .env
# Edit .env — set DATABASE_URL for your PostgreSQL instance
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Train the global model (once)
```
python train_global_model.py
```
Creates `./models/global_model.pkl`, `scaler.pkl`, `label_encoder.pkl`

### 4. Start FastAPI backend
```
uvicorn main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

### 5. Start Flask frontend
```
python app.py
```
UI: http://localhost:5000

---

## The only endpoint: `POST /agent-intake`

```json
{
  "patient_id":         "patient_919876543210",
  "name":               "Alice",
  "air_sensation":      2.5,
  "nasal_dryness":      3.0,
  "nasal_burning":      2.0,
  "suffocation":        2.5,
  "anxiety_score":      3.5,
  "humidity_level_pct": 62.0,
  "sleep_quality_hrs":  6.5,
  "signal_actual":      null
}
```

**Response:**
```json
{
  "patient_id":       "patient_919876543210",
  "reading_id":       1,
  "is_new_patient":   true,
  "signal_predicted": "Green",
  "ml_signal":        "Green",
  "z_override":       false,
  "max_abs_z":        0.0,
  "z_scores":         { "Air_Sensation": 0.0, ... },
  "top_deviations":   [...],
  "prob_green":       0.8542,
  "prob_yellow":      0.1021,
  "prob_red":         0.0437,
  "confidence":       0.8542,
  "model_used":       "global",
  "borderline":       false,
  "baseline_updated": true,
  "signal_actual":    null,
  "z_summary":        "All clinical features are within your normal range."
}
```

---

## Data flow

```
Retell call ends
       │
       ▼
app.py extracts [DATA:{...}] from transcript
       │
       ▼  POST /agent-intake
FastAPI main.py
  ├─ auto-register if new patient
  ├─ log daily reading (Z-score deviations vs Welford baseline)
  ├─ update Welford baseline
  ├─ auto-train personal model if ≥30 labeled readings
  └─ predict: personal model → global model → Z-score override
       │
       ▼
PostgreSQL (4 tables)
  patient_profiles · daily_readings · predictions · model_metadata
```

---

## Database tables

| Table | Purpose |
|---|---|
| `patient_profiles` | One row/patient; stores Welford baseline JSON |
| `daily_readings` | All 7 clinical features + Z-score deviations per reading |
| `predictions` | ML prediction history |
| `model_metadata` | Personal model training log |

---

## Validation rules

| Field | Range |
|---|---|
| air_sensation, nasal_dryness, nasal_burning, suffocation, anxiety_score | 1–10 |
| humidity_level_pct | 0–100 |
| sleep_quality_hrs | 0–12 |
| signal_actual | Green \| Yellow \| Red \| null |

---

## Run tests
```
python test_flow.py
```
