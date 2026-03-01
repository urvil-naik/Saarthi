from __future__ import annotations
"""
Aria — Daily Health Calling Agent
Flask frontend + Retell AI integration

Pipeline:
  Retell call ends
    → transcript captured
    → Claude AI analyses emotion/intensity → 7 clinical scores (JSON)
    → POST /agent-intake to FastAPI
    → Stored in PostgreSQL across 4 tables
"""

import os
import re
import json
import hmac
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from retell import Retell
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
RETELL_API_KEY    = os.environ.get("RETELL_API_KEY", "")
RETELL_AGENT_ID   = os.environ.get("RETELL_AGENT_ID", "")
FROM_NUMBER       = os.environ.get("FROM_NUMBER", "")
WEBHOOK_SECRET    = os.environ.get("RETELL_WEBHOOK_SECRET", "")
ENS_API_URL       = os.environ.get("ENS_API_URL", "http://localhost:8000")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

retell_client = Retell(api_key=RETELL_API_KEY)

# In-memory store (keyed by call_id)
calls_store: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/call", methods=["POST"])
def initiate_call():
    data         = request.get_json(force=True)
    to_number    = data.get("phone_number", "").strip()
    patient_name = data.get("patient_name", "there").strip()

    if not to_number:
        return jsonify({"error": "Phone number is required"}), 400

    to_number = re.sub(r"[\s\-()]", "", to_number)
    if not to_number.startswith("+"):
        to_number = "+1" + to_number

    if not RETELL_API_KEY or not RETELL_AGENT_ID or not FROM_NUMBER:
        return jsonify({"error": "Server not configured — check .env (RETELL_API_KEY, RETELL_AGENT_ID, FROM_NUMBER)"}), 500

    try:
        call = retell_client.call.create_phone_call(
            from_number=FROM_NUMBER,
            to_number=to_number,
            override_agent_id=RETELL_AGENT_ID,
            retell_llm_dynamic_variables={"patient_name": patient_name},
        )
        call_id = call.call_id
        calls_store[call_id] = {
            "call_id":          call_id,
            "to_number":        to_number,
            "patient_name":     patient_name,
            "patient_id":       _phone_to_patient_id(to_number),
            "status":           "initiated",
            "transcript":       [],
            "transcript_text":  "",
            "started_at":       datetime.utcnow().isoformat(),
            "ended_at":         None,
            "ens_result":       None,
            "ens_error":        None,
            "extracted_scores": None,
        }
        print(f"[CALL STARTED] call_id={call_id}  to={to_number}  patient={patient_name}")
        return jsonify({"success": True, "call_id": call_id})

    except Exception as e:
        print(f"[ERROR initiating call] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/call/<call_id>", methods=["GET"])
def get_call(call_id):
    try:
        call          = retell_client.call.retrieve(call_id)
        retell_status = call.call_status
        print(f"[POLL] call_id={call_id}  retell_status={retell_status}")

        if call_id not in calls_store:
            calls_store[call_id] = {
                "call_id":          call_id,
                "status":           "initiated",
                "transcript":       [],
                "transcript_text":  "",
                "started_at":       None,
                "ended_at":         None,
                "ens_result":       None,
                "ens_error":        None,
                "extracted_scores": None,
            }

        store = calls_store[call_id]

        if retell_status == "ongoing":
            store["status"] = "in_progress"

        elif retell_status in ("ended", "error") and store["status"] != "completed":
            transcript_objects = call.transcript_object or []
            transcript_raw     = call.transcript or ""
            print(f"[TRANSCRIPT] objects={len(transcript_objects)}  raw_len={len(transcript_raw)}")

            structured      = _convert_transcript_objects(transcript_objects, transcript_raw)
            transcript_text = _build_transcript_text(structured)

            store["status"]         = "completed"
            store["ended_at"]       = datetime.utcnow().isoformat()
            store["transcript"]     = structured
            store["transcript_text"] = transcript_text
            print(f"[COMPLETED] {len(structured)} messages stored")

            if store.get("ens_result") is None and store.get("ens_error") is None:
                _submit_to_ens(call_id, transcript_text)

        return jsonify(store)

    except Exception as e:
        print(f"[ERROR polling {call_id}] {e}")
        if call_id in calls_store:
            return jsonify(calls_store[call_id])
        return jsonify({"error": str(e)}), 404


@app.route("/webhook/retell", methods=["POST"])
def retell_webhook():
    if WEBHOOK_SECRET:
        sig      = request.headers.get("X-Retell-Signature", "")
        body     = request.get_data()
        expected = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return jsonify({"error": "Invalid signature"}), 401

    payload   = request.get_json(force=True)
    event     = payload.get("event", "")
    call_data = payload.get("data", {})
    call_id   = call_data.get("call_id", "")
    print(f"[WEBHOOK] event={event}  call_id={call_id}")

    if not call_id:
        return jsonify({"received": True}), 200

    if event == "call_started":
        if call_id not in calls_store:
            calls_store[call_id] = {
                "call_id":          call_id,
                "status":           "in_progress",
                "transcript":       [],
                "transcript_text":  "",
                "started_at":       datetime.utcnow().isoformat(),
                "ended_at":         None,
                "ens_result":       None,
                "ens_error":        None,
                "extracted_scores": None,
            }
        else:
            calls_store[call_id]["status"] = "in_progress"

    elif event == "call_ended":
        transcript_raw = call_data.get("transcript", "")
        transcript_obj = call_data.get("transcript_object", [])

        if call_id not in calls_store:
            calls_store[call_id] = {
                "call_id":          call_id,
                "status":           "completed",
                "transcript":       [],
                "transcript_text":  "",
                "started_at":       None,
                "ended_at":         datetime.utcnow().isoformat(),
                "ens_result":       None,
                "ens_error":        None,
                "extracted_scores": None,
            }

        if transcript_obj and isinstance(transcript_obj[0], dict):
            structured = [
                {"role": m.get("role", "unknown"), "content": m.get("content", "")}
                for m in transcript_obj
            ]
        else:
            structured = _parse_raw_transcript(transcript_raw)

        transcript_text = _build_transcript_text(structured)
        store = calls_store[call_id]
        store["status"]          = "completed"
        store["ended_at"]        = datetime.utcnow().isoformat()
        store["transcript"]      = structured
        store["transcript_text"] = transcript_text
        print(f"[WEBHOOK COMPLETED] {len(structured)} messages")

        if store.get("ens_result") is None and store.get("ens_error") is None:
            _submit_to_ens(call_id, transcript_text)

    return jsonify({"received": True}), 200


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE AI TRANSCRIPT ANALYSIS
# Converts raw call transcript → 7 clinical scores using emotion & intensity
# understanding — no [DATA:...] marker needed in the transcript.
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_PROMPT = """You are a clinical data extraction AI for an ENS (Environmental Nasal Sensitivity) health monitoring system.

Your job: read a health call transcript and output 7 clinical scores as JSON.

SCORING RULES:
- Air_Sensation:      How bad is breathing/air quality? (1=perfect, 10=unbearable)
- Nasal_Dryness:      Nasal dryness severity (1=moist/fine, 10=cracked/extremely dry)
- Nasal_Burning:      Burning in nose/throat (1=none, 10=severe constant burning)
- Suffocation:        Breathlessness/chest tightness (1=breathing freely, 10=can barely breathe)
- Anxiety_Score:      Overall anxiety from tone + content (1=calm, 10=extremely anxious)
- Humidity_Level_Pct: Indoor humidity % if mentioned (0-100, default 50 if unknown)
- Sleep_Quality_Hrs:  Hours of sleep last night (0-12, default 7 if not mentioned)

INTENSITY → SCORE MAPPING:
"fine/good/great/no problems"     → 1-2
"little bit/mild/slight"          → 3-4
"somewhat/noticeable/moderate"    → 5-6
"quite bad/a lot/significant"     → 7-8
"terrible/unbearable/can't breathe/horrible" → 9-10

TONE SIGNALS (adjust Anxiety_Score):
- Slow speech, sighing, pausing, voice breaking → add 1-2 points
- Fast worried speech, multiple complaints → add 1-2 points
- Calm, measured, matter-of-fact tone → subtract 1 point

If the patient never mentions a specific symptom → use 3 (mild) as default.
If patient says everything is fine/good → use 2 for that symptom.

TRANSCRIPT:
{transcript}

Respond with ONLY valid JSON. No explanation, no markdown fences:
{{"Air_Sensation": <1-10>, "Nasal_Dryness": <1-10>, "Nasal_Burning": <1-10>, "Suffocation": <1-10>, "Anxiety_Score": <1-10>, "Humidity_Level_Pct": <0-100>, "Sleep_Quality_Hrs": <0-12>, "reasoning": "<one sentence summarising key signals found>"}}"""


def _analyze_transcript_with_claude(transcript_text: str) -> dict | None:
    """
    Use Claude Haiku to analyze transcript and extract 7 clinical scores.
    Falls back to keyword-based heuristics if no ANTHROPIC_API_KEY set.
    """
    if not ANTHROPIC_API_KEY:
        print("[ANALYSIS] No ANTHROPIC_API_KEY found — using keyword fallback")
        return _keyword_fallback(transcript_text)

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "messages": [{
                    "role":    "user",
                    "content": ANALYSIS_PROMPT.format(transcript=transcript_text),
                }],
            },
            timeout=30,
        )
        resp.raise_for_status()

        raw = resp.json()["content"][0]["text"].strip()
        print(f"[ANALYSIS] Claude raw: {raw[:300]}")

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())

        scores = json.loads(raw)
        result = _validate_and_clamp(scores)
        print(f"[ANALYSIS] Extracted scores: { {k:v for k,v in result.items() if k != 'reasoning'} }")
        print(f"[ANALYSIS] Reasoning: {result.get('reasoning','')}")
        return result

    except json.JSONDecodeError as e:
        print(f"[ANALYSIS] JSON parse error ({e}) — falling back to keyword extraction")
        return _keyword_fallback(transcript_text)
    except Exception as e:
        print(f"[ANALYSIS] Claude API error: {e} — falling back to keyword extraction")
        return _keyword_fallback(transcript_text)


def _validate_and_clamp(scores: dict) -> dict:
    """Ensure all 7 keys exist and values are within legal ranges."""
    return {
        "Air_Sensation":      _clamp(scores.get("Air_Sensation",      3), lo=1,  hi=10),
        "Nasal_Dryness":      _clamp(scores.get("Nasal_Dryness",      3), lo=1,  hi=10),
        "Nasal_Burning":      _clamp(scores.get("Nasal_Burning",      3), lo=1,  hi=10),
        "Suffocation":        _clamp(scores.get("Suffocation",         3), lo=1,  hi=10),
        "Anxiety_Score":      _clamp(scores.get("Anxiety_Score",       3), lo=1,  hi=10),
        "Humidity_Level_Pct": _clamp(scores.get("Humidity_Level_Pct", 50), lo=0,  hi=100),
        "Sleep_Quality_Hrs":  _clamp(scores.get("Sleep_Quality_Hrs",   7), lo=0,  hi=12),
        "reasoning":          str(scores.get("reasoning", "Extracted via AI analysis")),
    }


def _keyword_fallback(transcript_text: str) -> dict:
    """
    Keyword + intensity heuristic fallback when Claude API is unavailable.
    Scans only the patient's lines for symptom signals.
    """
    print("[ANALYSIS] Running keyword fallback extraction")

    # Extract patient-only lines for symptom scoring
    patient_lines = []
    for line in transcript_text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("patient:"):
            patient_lines.append(stripped[8:].strip().lower())
    pt = " ".join(patient_lines) if patient_lines else transcript_text.lower()

    def score_from_keywords(levels: dict, default: float = 3.0) -> float:
        result = default
        for score_val, words in sorted(levels.items()):
            for w in words:
                if w in pt:
                    result = float(score_val)
        return result

    air = score_from_keywords({
        2: ["air is fine", "breathing fine", "air feels good", "air is good", "fresh air"],
        4: ["little stuffy", "slight stuffiness", "bit stuffy"],
        6: ["stuffy", "dusty", "polluted", "not great air", "air quality bad"],
        8: ["very stuffy", "hard to breathe", "air is terrible", "can barely breathe"],
        9: ["suffocating air", "no fresh air", "air is unbearable"],
    })

    dryness = score_from_keywords({
        2: ["nose is fine", "not dry", "no dryness", "moist", "nose feels fine"],
        3: ["little dry", "slightly dry", "bit dry", "minor dryness"],
        6: ["dry", "quite dry", "dryness", "my nose is dry"],
        8: ["very dry", "really dry", "extremely dry", "cracked", "painful dryness"],
        9: ["nose is cracking", "bleeding", "unbearably dry"],
    })

    burning = score_from_keywords({
        1: ["no burning", "no irritation", "no sting", "no pain in nose"],
        3: ["little burning", "slight burn", "mild irritation"],
        6: ["burning", "irritation", "stinging", "sting", "burn"],
        8: ["bad burning", "bad irritation", "intense burning", "really burns"],
        9: ["terrible burning", "unbearable burn", "constant burning"],
    }, default=2.0)

    suffocation = score_from_keywords({
        1: ["breathing perfectly", "no breathing issues", "lungs are fine"],
        3: ["little tight", "slight tightness", "chest feels slight"],
        6: ["breathless", "chest tight", "tightness", "hard to breathe"],
        8: ["very breathless", "severe tightness", "can barely breathe"],
        9: ["suffocating", "gasping", "no air"],
    }, default=2.0)

    anxiety = score_from_keywords({
        2: ["calm", "relaxed", "fine overall", "doing okay", "no stress"],
        4: ["little worried", "bit anxious", "slightly stressed"],
        6: ["worried", "anxious", "stressed", "concerned", "not doing well"],
        8: ["very anxious", "really stressed", "scared", "panicking"],
        9: ["extremely anxious", "terrified", "panicking badly"],
    })

    # Sleep — look for explicit hours first, then keywords
    sleep = 7.0
    sleep_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s+)?(?:sleep|sleeping)", pt)
    if sleep_match:
        sleep = _clamp(float(sleep_match.group(1)), lo=0, hi=12)
    elif any(w in pt for w in ["couldn't sleep", "no sleep", "didn't sleep", "barely slept"]):
        sleep = 2.0
    elif any(w in pt for w in ["bad sleep", "poor sleep", "rough night", "terrible sleep"]):
        sleep = 4.0
    elif any(w in pt for w in ["good sleep", "slept well", "great sleep", "slept fine"]):
        sleep = 8.0

    # Humidity — look for explicit % first, then keywords
    humidity = 50.0
    hum_match = re.search(r"(\d+)\s*(?:%|percent)\s*(?:humidity)?", pt)
    if hum_match:
        humidity = _clamp(float(hum_match.group(1)), lo=0, hi=100)
    elif any(w in pt for w in ["very humid", "humid inside"]):
        humidity = 72.0
    elif any(w in pt for w in ["dry air", "very dry air", "no moisture"]):
        humidity = 18.0

    return _validate_and_clamp({
        "Air_Sensation":      air,
        "Nasal_Dryness":      dryness,
        "Nasal_Burning":      burning,
        "Suffocation":        suffocation,
        "Anxiety_Score":      anxiety,
        "Humidity_Level_Pct": humidity,
        "Sleep_Quality_Hrs":  sleep,
        "reasoning":          "Extracted via keyword/intensity fallback (add ANTHROPIC_API_KEY for AI analysis)",
    })


# ══════════════════════════════════════════════════════════════════════════════
# ENS SUBMISSION → FastAPI → PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════

def _phone_to_patient_id(phone: str) -> str:
    """'+919876543210' → 'patient_919876543210'"""
    return "patient_" + re.sub(r"\D", "", phone)


def _clamp(val: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, float(val)))


def _submit_to_ens(call_id: str, transcript_text: str) -> None:
    """
    Full pipeline:
      1. Validate transcript is not empty.
      2. Analyze with Claude AI (or keyword fallback).
      3. POST structured 7-score payload to FastAPI /agent-intake.
      4. FastAPI stores data across 4 PostgreSQL tables (see below).
      5. Store result back in calls_store for UI polling.

    What gets stored in PostgreSQL per call:
      patient_profiles  → patient identity + Welford baseline (JSON) + total_readings counter
      daily_readings    → all 7 clinical scores + Z-score deviations (JSON) + signal_actual
      predictions       → ML signal + prob_green/yellow/red + confidence + model_used
      model_metadata    → personal model training log (written when >=30 labeled readings)
    """
    store = calls_store.get(call_id)
    if not store:
        return

    if not transcript_text or not transcript_text.strip():
        store["ens_error"] = "Empty transcript — nothing to analyse."
        print(f"[ENS] {call_id}: empty transcript")
        return

    print(f"[ENS] {call_id}: analysing transcript ({len(transcript_text)} chars) ...")

    # Step 1 — AI analysis
    scores = _analyze_transcript_with_claude(transcript_text)
    if scores is None:
        store["ens_error"] = "Transcript analysis failed."
        return

    store["extracted_scores"] = scores

    patient_id   = store.get("patient_id") or _phone_to_patient_id(store.get("to_number", call_id))
    patient_name = store.get("patient_name") or None

    # Step 2 — Build /agent-intake payload
    payload = {
        "patient_id":         patient_id,
        "name":               patient_name,
        "air_sensation":      scores["Air_Sensation"],
        "nasal_dryness":      scores["Nasal_Dryness"],
        "nasal_burning":      scores["Nasal_Burning"],
        "suffocation":        scores["Suffocation"],
        "anxiety_score":      scores["Anxiety_Score"],
        "humidity_level_pct": scores["Humidity_Level_Pct"],
        "sleep_quality_hrs":  scores["Sleep_Quality_Hrs"],
        "signal_actual":      None,
    }
    print(f"[ENS] {call_id}: posting → {payload}")

    # Step 3 — POST to FastAPI
    try:
        resp = requests.post(
            f"{ENS_API_URL}/agent-intake",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        ens = resp.json()

        # Step 4 — Store result (FastAPI has already written to DB by this point)
        store["ens_result"] = {
            "patient_id":     ens.get("patient_id"),
            "is_new_patient": ens.get("is_new_patient"),
            "signal":         ens.get("signal_predicted"),
            "ml_signal":      ens.get("ml_signal"),
            "z_override":     ens.get("z_override"),
            "confidence":     ens.get("confidence"),
            "model_used":     ens.get("model_used"),
            "borderline":     ens.get("borderline"),
            "prob_green":     ens.get("prob_green"),
            "prob_yellow":    ens.get("prob_yellow"),
            "prob_red":       ens.get("prob_red"),
            "max_abs_z":      ens.get("max_abs_z"),
            "z_scores":       ens.get("z_scores", {}),
            "top_deviations": ens.get("top_deviations", []),
            "z_summary":      ens.get("z_summary"),
            "scores_used":    scores,
            "reasoning":      scores.get("reasoning", ""),
        }
        print(
            f"[ENS] {call_id}: signal={store['ens_result']['signal']}  "
            f"confidence={store['ens_result']['confidence']}"
        )

    except requests.exceptions.ConnectionError:
        store["ens_error"] = f"ENS API unreachable at {ENS_API_URL} — is uvicorn running on port 8000?"
    except requests.exceptions.HTTPError as e:
        store["ens_error"] = f"ENS API {e.response.status_code}: {e.response.text[:400]}"
    except Exception as e:
        store["ens_error"] = f"ENS error: {e}"

    if store.get("ens_error"):
        print(f"[ENS ERROR] {call_id}: {store['ens_error']}")


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _convert_transcript_objects(objects, raw_fallback: str) -> list:
    """
    Retell SDK returns objects with .role / .content attributes (not plain dicts).
    Falls back to raw text parsing if objects list is empty.
    """
    if not objects:
        return _parse_raw_transcript(raw_fallback)
    result = []
    for obj in objects:
        try:
            role    = obj.role    if hasattr(obj, "role")    else obj.get("role", "unknown")
            content = obj.content if hasattr(obj, "content") else obj.get("content", "")
        except Exception:
            continue
        content = (content or "").strip()
        if content:
            result.append({"role": role, "content": content})
    return result


def _parse_raw_transcript(raw: str) -> list:
    """Parse 'Agent: ...' / 'User: ...' style plain-text transcript."""
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("agent:"):
            lines.append({"role": "agent", "content": line[6:].strip()})
        elif lower.startswith("user:"):
            lines.append({"role": "user",  "content": line[5:].strip()})
        else:
            lines.append({"role": "agent", "content": line})
    return lines


def _build_transcript_text(structured: list) -> str:
    lines = []
    for msg in structured:
        role    = "Aria" if msg["role"] == "agent" else "Patient"
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
