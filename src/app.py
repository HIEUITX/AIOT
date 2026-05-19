# src/app.py

from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

FEATURES = [
    "value",
    "rolling_mean",
    "rolling_std",
    "delta",
    "zscore",
    "stuck"
]

model = joblib.load(
    "models/isolation_forest_iforest_v1.joblib"
)

@app.get("/health")
def health():

    return {
        "model_loaded": True,
        "model_type": "IsolationForest"
    }

@app.get("/model-info")
def model_info():

    return {
        "model_version": "iforest_v1",
        "features": FEATURES
    }

@app.post("/detect-anomaly")
def detect_anomaly(data: dict):

    df = pd.DataFrame([data])

    score = -model.decision_function(df)[0]

    is_anomaly = bool(score > 0.5)

    if score > 0.8:
        severity = "HIGH"

    elif score > 0.5:
        severity = "MEDIUM"

    else:
        severity = "LOW"

    if severity == "HIGH":
        decision = (
            "CREATE_ALERT_AND_REQUIRE_HUMAN_CHECK"
        )

    elif severity == "MEDIUM":
        decision = "CHECK_SENSOR"

    else:
        decision = "MONITOR"

    return {

        "model_output": {

            "anomaly_score": float(score),

            "threshold_used": 0.5,

            "is_anomaly": is_anomaly,

            "model_version": "iforest_v1"
        },

        "event": {

            "event_type":
            "TEMPERATURE_PATTERN_DEVIATION",

            "severity": severity,

            "decision": decision,

            "explanation":
            "temperature deviates strongly from recent pattern",

            "safety_note":
            "Không tự động điều khiển thiết bị khi anomaly cao"
        }
    }