# src/train_anomaly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================
# CREATE FOLDERS
# =========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("START LAB 3")

# =========================================
# LOAD DATA
# =========================================

DATA_PATH = "data/ambient_temperature_system_failure.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATA HEAD =====")
print(df.head())

print("\nTOTAL ROWS:", len(df))

# =========================================
# TIME
# =========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# =========================================
# FEATURE ENGINEERING
# =========================================

# rolling mean
df["rolling_mean"] = (
    df["value"]
    .rolling(window=10)
    .mean()
)

# rolling std
df["rolling_std"] = (
    df["value"]
    .rolling(window=10)
    .std()
)

# delta
df["delta"] = (
    df["value"]
    .diff()
)

# z-score
df["zscore"] = (
    (
        df["value"]
        - df["value"].mean()
    )
    /
    df["value"].std()
)

# sensor stuck flag
df["stuck"] = (
    df["delta"].abs() < 0.0001
).astype(int)

# remove NaN
df = df.dropna()

print("\nFEATURE ENGINEERING DONE")

# =========================================
# VISUALIZE RAW DATA
# =========================================

plt.figure(figsize=(15, 5))

plt.plot(
    df["timestamp"],
    df["value"]
)

plt.title("Temperature Time Series")

plt.xlabel("Timestamp")
plt.ylabel("Temperature")

plt.savefig(
    "figures/raw_timeseries.png"
)

plt.close()

print("RAW TIMESERIES SAVED")

# =========================================
# TRAIN TEST SPLIT
# =========================================

train_size = int(len(df) * 0.7)

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print("\n===== TRAIN TEST =====")
print("TRAIN SIZE:", len(train_df))
print("TEST SIZE:", len(test_df))

# =========================================
# FEATURE LIST
# =========================================

features = [
    "value",
    "rolling_mean",
    "rolling_std",
    "delta",
    "zscore",
    "stuck"
]

# =========================================
# TRAIN MODEL
# =========================================

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(
    train_df[features]
)

print("\nMODEL TRAINED")

# =========================================
# PREDICT
# =========================================

scores = model.decision_function(
    test_df[features]
)

preds = model.predict(
    test_df[features]
)

# convert
# -1 => anomaly
# 1 => normal

test_df["is_anomaly"] = (
    preds == -1
).astype(int)

# anomaly score
test_df["anomaly_score"] = -scores

print("\nPREDICTION DONE")

# =========================================
# THRESHOLD
# =========================================

threshold = (
    test_df["anomaly_score"]
    .quantile(0.95)
)

print("THRESHOLD:", threshold)

# =========================================
# SEVERITY
# =========================================

def get_severity(score):

    if score > 0.8:
        return "HIGH"

    elif score > 0.5:
        return "MEDIUM"

    else:
        return "LOW"

test_df["severity"] = (
    test_df["anomaly_score"]
    .apply(get_severity)
)

# =========================================
# DECISION
# =========================================

def get_decision(severity):

    if severity == "HIGH":
        return (
            "CREATE_ALERT_AND_REQUIRE_HUMAN_CHECK"
        )

    elif severity == "MEDIUM":
        return "CHECK_SENSOR"

    else:
        return "MONITOR"

test_df["decision"] = (
    test_df["severity"]
    .apply(get_decision)
)

# =========================================
# EVENT TYPE
# =========================================

test_df["event_type"] = (
    "TEMPERATURE_PATTERN_DEVIATION"
)

# =========================================
# EXPLANATION
# =========================================

def get_explanation(row):

    if row["severity"] == "HIGH":

        return (
            "temperature deviates strongly "
            "from recent pattern"
        )

    elif row["severity"] == "MEDIUM":

        return (
            "possible abnormal "
            "temperature behavior"
        )

    else:

        return (
            "normal telemetry pattern"
        )

test_df["explanation"] = (
    test_df.apply(
        get_explanation,
        axis=1
    )
)

# =========================================
# SAFETY NOTE
# =========================================

test_df["safety_note"] = (
    "Không tự động điều khiển thiết bị khi anomaly cao"
)

# =========================================
# CREATE DEMO LABEL
# =========================================

test_df["y_true"] = (
    test_df["anomaly_score"] > threshold
).astype(int)

y_true = test_df["y_true"]

y_pred = test_df["is_anomaly"]

# =========================================
# METRICS
# =========================================

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)

metrics = {

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "threshold": float(threshold),

    "confusion_matrix": cm.tolist()
}

print("\n===== METRICS =====")
print(metrics)

# =========================================
# SAVE METRICS
# =========================================

with open(
    "outputs/iforest_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )

print("METRICS SAVED")

# =========================================
# SAVE PREDICTIONS
# =========================================

test_df.to_csv(
    "outputs/iforest_test_predictions.csv",
    index=False
)

print("PREDICTIONS SAVED")

# =========================================
# SAVE EVENT LOG
# =========================================

event_log = test_df[
    [
        "timestamp",
        "value",
        "anomaly_score",
        "is_anomaly",
        "severity",
        "decision",
        "event_type",
        "explanation",
        "safety_note"
    ]
]

event_log.to_csv(
    "outputs/anomaly_event_log.csv",
    index=False
)

print("EVENT LOG SAVED")

# =========================================
# SAVE MODEL
# =========================================

joblib.dump(
    model,
    "models/isolation_forest_iforest_v1.joblib"
)

print("MODEL SAVED")

# =========================================
# PLOT ANOMALY RESULT
# =========================================

plt.figure(figsize=(15, 5))

plt.plot(
    test_df["timestamp"],
    test_df["value"],
    label="Temperature"
)

anomalies = test_df[
    test_df["is_anomaly"] == 1
]

plt.scatter(
    anomalies["timestamp"],
    anomalies["value"]
)

plt.title("Anomaly Detection Result")

plt.xlabel("Timestamp")
plt.ylabel("Temperature")

plt.legend()

plt.savefig(
    "figures/anomaly_detection_result.png"
)

plt.close()

print("ANOMALY RESULT FIGURE SAVED")

# =========================================
# PLOT ANOMALY SCORE
# =========================================

plt.figure(figsize=(15, 5))

plt.plot(
    test_df["timestamp"],
    test_df["anomaly_score"],
    label="Anomaly Score"
)

plt.axhline(
    y=threshold,
    linestyle="--"
)

plt.title("Anomaly Score Over Time")

plt.xlabel("Timestamp")
plt.ylabel("Anomaly Score")

plt.legend()

plt.savefig(
    "figures/anomaly_score_over_time.png"
)

plt.close()

print("ANOMALY SCORE FIGURE SAVED")

# =========================================
# FINAL
# =========================================

print("\n=================================")
print("LAB 3 COMPLETED SUCCESSFULLY")
print("=================================")