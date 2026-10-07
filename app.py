
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy.stats import kurtosis, skew

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# ============================================================
# PUMPGUARD AI v2 — INDUSTRIAL / SCI-FI UI
# ============================================================
st.set_page_config(
    page_title="PumpGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Industrial dark UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(0, 180, 255, 0.10), transparent 24%),
        radial-gradient(circle at 15% 95%, rgba(0, 255, 200, 0.05), transparent 22%),
        #070b11;
    color: #e8f0f7;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: 0.3px;
}

h1 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1018 0%, #070b11 100%);
    border-right: 1px solid rgba(0, 210, 255, 0.18);
}

[data-testid="stSidebar"] * {
    color: #dce8f2;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0e1621, #0a1119);
    border: 1px solid rgba(0, 210, 255, 0.18);
    border-radius: 14px;
    padding: 12px 14px;
}

[data-testid="stMetricLabel"] {
    color: #8fa7b9 !important;
}

[data-testid="stMetricValue"] {
    color: #eaf7ff !important;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid rgba(0, 210, 255, 0.35);
    background: linear-gradient(90deg, #0d2633, #0b1721);
    color: #dff8ff;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #00d8ff;
    color: white;
    box-shadow: 0 0 18px rgba(0, 216, 255, 0.15);
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0, 210, 255, 0.35);
    border-radius: 12px;
    background: rgba(8, 18, 28, 0.6);
}

.panel {
    background: linear-gradient(145deg, rgba(15,24,35,.97), rgba(8,14,22,.97));
    border: 1px solid rgba(0, 210, 255, 0.16);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 35px rgba(0,0,0,.18);
}

.panel-title {
    font-family: 'Orbitron', sans-serif;
    color: #d9f7ff;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.topbar {
    border: 1px solid rgba(0, 210, 255, 0.20);
    background: linear-gradient(90deg, rgba(9,18,28,.96), rgba(10,23,32,.90));
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 18px;
}

.brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #effcff;
    letter-spacing: 1px;
}

.subtitle {
    color: #8199aa;
    font-size: 12px;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #00e5a8;
    box-shadow: 0 0 12px #00e5a8;
    margin-right: 7px;
}

.status-dot.off {
    background: #ffb020;
    box-shadow: 0 0 10px #ffb020;
}

.core-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,.06);
    padding: 8px 0;
    font-size: 12px;
}

.core-row:last-child { border-bottom: none; }

.core-name { color: #8098aa; }
.core-state { color: #00e5a8; font-weight: 600; }

.health-ring {
    width: 170px;
    height: 170px;
    border-radius: 50%;
    margin: 10px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: conic-gradient(#00d8ff 0deg, #00d8ff var(--scoredeg), #152432 var(--scoredeg), #152432 360deg);
    box-shadow: 0 0 28px rgba(0,216,255,.10);
}

.health-inner {
    width: 132px;
    height: 132px;
    border-radius: 50%;
    background: #09111a;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.health-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 32px;
    color: #f0fbff;
}

.health-label {
    color: #7992a5;
    font-size: 11px;
    letter-spacing: 1px;
}

.fault-card {
    border: 1px solid rgba(255, 176, 32, .32);
    background: linear-gradient(145deg, rgba(63,45,12,.38), rgba(25,20,10,.5));
    border-radius: 14px;
    padding: 18px;
}

.fault-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    color: #ffd27a;
    margin: 6px 0;
}

.confidence {
    color: #9bb1c1;
    font-size: 12px;
}

.sensor-card {
    background: #0a131d;
    border: 1px solid rgba(0, 210, 255, .12);
    border-radius: 12px;
    padding: 13px;
    text-align: center;
}

.sensor-name {
    color: #718b9e;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sensor-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 21px;
    color: #e9faff;
    margin-top: 5px;
}

.sensor-unit {
    color: #6e899b;
    font-size: 10px;
}

.digital-twin {
    background: linear-gradient(145deg, #09131c, #071018);
    border: 1px solid rgba(0, 210, 255, .18);
    border-radius: 16px;
    padding: 8px;
    text-align: center;
}

.twin-caption {
    color: #7892a4;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 5px;
}

.tag {
    display: inline-block;
    border: 1px solid rgba(0, 229, 168, .3);
    background: rgba(0,229,168,.06);
    color: #00e5a8;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 10px;
    margin-right: 4px;
}

.small-note {
    color: #728b9e;
    font-size: 11px;
}

.ai-box {
    border-left: 3px solid #00d8ff;
    background: rgba(0,216,255,.045);
    padding: 12px 14px;
    border-radius: 0 10px 10px 0;
    color: #cfe9f2;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,210,255,.12);
    border-radius: 10px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================
DB_FILE = "pump_history.db"

def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            fault TEXT,
            confidence REAL,
            health_score REAL,
            vibration_rms REAL,
            peak REAL,
            peak_to_peak REAL,
            std REAL,
            kurtosis REAL,
            skewness REAL,
            crest_factor REAL,
            dominant_frequency REAL,
            temperature REAL,
            current REAL,
            rpm REAL,
            source TEXT
        )
    """)
    con.commit()
    con.close()

def save_history(result):
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        INSERT INTO diagnosis_history
        (timestamp, fault, confidence, health_score, vibration_rms, peak,
         peak_to_peak, std, kurtosis, skewness, crest_factor,
         dominant_frequency, temperature, current, rpm, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        result.get("fault", "Unknown"),
        result.get("confidence", np.nan),
        result.get("health_score", np.nan),
        result.get("RMS", np.nan),
        result.get("Peak", np.nan),
        result.get("Peak-to-Peak", np.nan),
        result.get("Std", np.nan),
        result.get("Kurtosis", np.nan),
        result.get("Skewness", np.nan),
        result.get("Crest Factor", np.nan),
        result.get("Dominant Frequency", np.nan),
        result.get("Temperature", np.nan),
        result.get("Current", np.nan),
        result.get("RPM", np.nan),
        result.get("source", "CSV")
    ))
    con.commit()
    con.close()

def load_history():
    con = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM diagnosis_history ORDER BY id DESC", con
    )
    con.close()
    return df


# ============================================================
# SIGNAL PROCESSING
# ============================================================
def clean_numeric_signal(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values.astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0])
    return x - np.mean(x)

def fft_features(signal, fs):
    signal = clean_numeric_signal(signal)
    n = len(signal)

    if n < 8:
        return None, None, {}

    window = np.hanning(n)
    y = signal * window
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    idx = np.argmax(spectrum[1:]) + 1 if len(spectrum) > 1 else 0

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    p2p = float(np.ptp(signal))
    std = float(np.std(signal))
    kurt = float(kurtosis(signal, fisher=False, bias=False)) if n > 3 else 0.0
    sk = float(skew(signal, bias=False)) if n > 2 else 0.0
    crest = float(peak / rms) if rms > 0 else 0.0
    dom = float(freqs[idx])

    features = {
        "RMS": rms,
        "Peak": peak,
        "Peak-to-Peak": p2p,
        "Std": std,
        "Kurtosis": kurt,
        "Skewness": sk,
        "Crest Factor": crest,
        "Dominant Frequency": dom
    }
    return freqs, spectrum, features


# ============================================================
# DATA HANDLING
# ============================================================
ALIASES = {
    "vibration": ["vibration", "acceleration", "amplitude", "signal", "vibration_signal", "accel"],
    "temperature": ["temperature", "temp", "temperature_c", "temp_c"],
    "current": ["current", "motor_current", "current_a", "amps", "ampere"],
    "rpm": ["rpm", "speed", "motor_speed"],
    "label": ["fault", "fault_type", "label", "class", "condition", "status"]
}

BASE_FEATURES = [
    "RMS", "Peak", "Peak-to-Peak", "Std", "Kurtosis",
    "Skewness", "Crest Factor", "Dominant Frequency",
    "Temperature", "Current", "RPM"
]

def find_column(df, names):
    lower = {str(c).strip().lower(): c for c in df.columns}

    for n in names:
        if n in lower:
            return lower[n]

    for c in df.columns:
        cl = str(c).strip().lower().replace(" ", "_")
        if any(n in cl for n in names):
            return c

    return None

def standardize_columns(df):
    return {
        key: find_column(df, aliases)
        for key, aliases in ALIASES.items()
    }

def available_features(df):
    return [
        c for c in BASE_FEATURES
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

def prepare_training_table(df, fs):
    found = standardize_columns(df)
    label_col = found["label"]

    if label_col is None:
        return None, "No Fault/Label column found."

    raw_col = found["vibration"]

    # Check whether uploaded file already contains extracted features.
    existing = {}
    for f in BASE_FEATURES:
        c = find_column(df, [
            f.lower(),
            f.lower().replace("-", "_"),
            f.lower().replace(" ", "_")
        ])
        existing[f] = c

    if sum(v is not None for v in existing.values()) >= 5:
        out = pd.DataFrame(index=df.index)

        for f, c in existing.items():
            if c is not None:
                out[f] = pd.to_numeric(df[c], errors="coerce")

        for sensor_key, col in [
            ("Temperature", found["temperature"]),
            ("Current", found["current"]),
            ("RPM", found["rpm"])
        ]:
            if col is not None:
                out[sensor_key] = pd.to_numeric(df[col], errors="coerce")

        out["Fault"] = df[label_col].astype(str)
        return out.dropna(subset=["Fault"]), None

    # Raw vibration input.
    if raw_col is None:
        return None, (
            "The CSV does not contain enough recognized feature columns "
            "and no raw vibration/signal column was detected."
        )

    signal = clean_numeric_signal(df[raw_col])
    _, _, feats = fft_features(signal, fs)

    if not feats:
        return None, "Vibration signal is too short for FFT."

    for sensor_key, col in [
        ("Temperature", found["temperature"]),
        ("Current", found["current"]),
        ("RPM", found["rpm"])
    ]:
        if col is not None:
            feats[sensor_key] = float(
                pd.to_numeric(df[col], errors="coerce").median()
            )

    label = df[label_col].dropna().astype(str).mode()
    feats["Fault"] = label.iloc[0] if len(label) else "Unknown"

    return pd.DataFrame([feats]), None


# ============================================================
# DEMO DATA
# ============================================================
def make_demo_dataset(n_per_class=160, seed=42):
    rng = np.random.default_rng(seed)

    specs = {
        "Healthy": {
            "RMS": (0.18, .04), "Peak": (.55, .10), "Peak-to-Peak": (1.05, .18),
            "Std": (.17, .035), "Kurtosis": (2.7, .35), "Skewness": (.02, .12),
            "Crest Factor": (3.0, .35), "Dominant Frequency": (50, 3),
            "Temperature": (34, 2), "Current": (2.2, .25), "RPM": (2910, 12)
        },
        "Misalignment": {
            "RMS": (.58, .10), "Peak": (1.55, .25), "Peak-to-Peak": (3.0, .40),
            "Std": (.52, .08), "Kurtosis": (3.8, .55), "Skewness": (.10, .18),
            "Crest Factor": (3.1, .40), "Dominant Frequency": (100, 5),
            "Temperature": (42, 3), "Current": (3.1, .35), "RPM": (2895, 18)
        },
        "Impeller Damage": {
            "RMS": (.82, .14), "Peak": (2.25, .35), "Peak-to-Peak": (4.4, .55),
            "Std": (.75, .12), "Kurtosis": (5.0, .8), "Skewness": (.25, .22),
            "Crest Factor": (3.35, .45), "Dominant Frequency": (298, 7),
            "Temperature": (47, 3), "Current": (3.5, .40), "RPM": (2888, 22)
        }
    }

    rows = []

    for cls, spec in specs.items():
        for _ in range(n_per_class):
            row = {
                k: rng.normal(mu, sd)
                for k, (mu, sd) in spec.items()
            }

            for k in [
                "RMS", "Peak", "Peak-to-Peak", "Std",
                "Kurtosis", "Crest Factor",
                "Dominant Frequency", "Current", "RPM"
            ]:
                row[k] = max(0.01, row[k])

            row["Fault"] = cls
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MACHINE LEARNING
# ============================================================
def train_models(df, selected_features):
    """
    Train a broader set of classification algorithms.

    Important:
    - Accuracy/precision/recall/F1 are calculated ONLY on the held-out test set.
    - 5-fold stratified CV is also reported to show robustness.
    - Scaling is applied inside pipelines for algorithms that need it.
    """
    X = df[selected_features].copy()
    y = df["Fault"].astype(str)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if y.nunique() < 2:
        raise ValueError("At least two fault classes are required.")

    if y.value_counts().min() < 5:
        raise ValueError("Each fault class needs at least 5 samples for 5-fold CV.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=.25,
        random_state=42,
        stratify=y
    )

    # Core engineering / ML algorithms.
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "Hist. Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                C=2.0,
                kernel="rbf",
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            ))
        ]),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.0005,
                max_iter=1200,
                early_stopping=True,
                random_state=42
            ))
        ]),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=180,
            learning_rate=0.05,
            random_state=42
        )
    }

    # A soft-voting ensemble combines diverse high-performing classifiers.
    voting_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=180, random_state=42, class_weight="balanced"
        )),
        ("extra", ExtraTreesClassifier(
            n_estimators=180, random_state=42, class_weight="balanced"
        )),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, C=2.0, class_weight="balanced",
                          random_state=42))
        ]))
    ]

    models["Soft Voting Ensemble"] = VotingClassifier(
        estimators=voting_estimators,
        voting="soft"
    )

    results = {}
    fitted = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Cross-validation is run on the training partition only.
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="accuracy"
        )

        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
            "cm": confusion_matrix(
                y_test, pred, labels=sorted(y.unique())
            ),
            "labels": sorted(y.unique())
        }

        fitted[name] = model

    best_name = max(
        results,
        key=lambda k: (
            results[k]["f1"],
            results[k]["cv_mean"],
            results[k]["accuracy"]
        )
    )

    return fitted, results, best_name, X_train, X_test, y_train, y_test


def cross_validate_model(model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    return scores


# ============================================================
# HEALTH + MAINTENANCE
# ============================================================
def health_score(fault, confidence, features):
    # Prototype index — replace with experimentally calibrated limits later.
    base = {
        "Healthy": 95,
        "Good": 85,
        "Misalignment": 62,
        "Impeller Damage": 50
    }

    score = base.get(
        fault,
        max(20, 85 - 50 * confidence)
    )

    if features.get("Temperature", np.nan) > 50:
        score -= 8

    if features.get("Current", np.nan) > 4:
        score -= 7

    if features.get("RMS", np.nan) > 1.2:
        score -= 8

    score += 5 * (confidence - .5)

    return float(np.clip(score, 0, 100))


def health_label(score):
    if score >= 80:
        return "HEALTHY"
    if score >= 60:
        return "MONITOR"
    if score >= 40:
        return "SERVICE REQUIRED"
    return "CRITICAL"


def recommendation(fault):
    return {
        "Healthy":
            "Continue normal operation and maintain routine vibration monitoring.",
        "Misalignment":
            "Inspect coupling and motor-pump shaft alignment. Check 1×/2× rotational components and re-align if required.",
        "Impeller Damage":
            "Inspect impeller vanes for damage, blockage or imbalance. Verify vibration spectrum before corrective action.",
        "Bearing Fault":
            "Inspect bearing condition, lubrication and mounting. Confirm characteristic bearing-frequency components.",
        "Rotor Unbalance":
            "Inspect rotor/impeller balance and check for material buildup or mechanical damage.",
        "Loose Foundation":
            "Inspect foundation bolts, baseplate and mounting stiffness.",
        "Unknown":
            "Collect more representative labelled experimental data before making a maintenance decision."
    }.get(
        fault,
        "Inspect the pump and validate the diagnosis with additional measurements."
    )


# ============================================================
# DIGITAL TWIN SVG
# ============================================================
def pump_svg(status="RUNNING"):
    return f"""
    <div class="digital-twin">
    <svg viewBox="0 0 520 260" width="100%" height="230">
        <defs>
          <linearGradient id="metal" x1="0" x2="1">
            <stop offset="0%" stop-color="#172936"/>
            <stop offset="50%" stop-color="#2b4657"/>
            <stop offset="100%" stop-color="#10202b"/>
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <!-- base -->
        <rect x="60" y="215" width="400" height="10" rx="5" fill="#12202a"/>

        <!-- motor -->
        <rect x="300" y="75" width="135" height="90" rx="16"
              fill="url(#metal)" stroke="#00d8ff" stroke-opacity=".55"/>
        <circle cx="335" cy="120" r="27" fill="none"
                stroke="#00d8ff" stroke-opacity=".65" stroke-width="4"/>
        <path d="M335 100 L335 140 M315 120 L355 120"
              stroke="#00d8ff" stroke-opacity=".6" stroke-width="3"/>
        <text x="367" y="112" fill="#a7c7d7" font-size="13">MOTOR</text>
        <text x="367" y="132" fill="#00e5a8" font-size="11">{status}</text>

        <!-- coupling -->
        <rect x="260" y="103" width="40" height="35" rx="5"
              fill="#253b49" stroke="#ffb020" stroke-opacity=".6"/>
        <line x1="255" y1="120" x2="300" y2="120"
              stroke="#ffb020" stroke-width="4"/>

        <!-- pump casing -->
        <circle cx="150" cy="120" r="72" fill="url(#metal)"
                stroke="#00d8ff" stroke-opacity=".65" stroke-width="3"/>
        <circle cx="150" cy="120" r="35" fill="#071018"
                stroke="#00d8ff" stroke-opacity=".45" stroke-width="3"/>
        <circle cx="150" cy="120" r="11" fill="#00e5a8"
                filter="url(#glow)"/>

        <!-- suction / discharge -->
        <rect x="35" y="103" width="43" height="34" rx="6"
              fill="#1c303e" stroke="#00d8ff" stroke-opacity=".5"/>
        <rect x="186" y="42" width="34" height="48" rx="6"
              fill="#1c303e" stroke="#00d8ff" stroke-opacity=".5"/>

        <text x="112" y="211" fill="#8ca6b7" font-size="12">CENTRIFUGAL PUMP</text>
        <text x="18" y="95" fill="#718b9e" font-size="9">SUCTION</text>
        <text x="188" y="35" fill="#718b9e" font-size="9">DISCHARGE</text>
    </svg>
    <div class="twin-caption">DIGITAL TWIN // CENTRIFUGAL PUMP</div>
    </div>
    """


# ============================================================
# OPTIONAL GENAI
# ============================================================
def genai_explanation(fault, confidence, score, features, shap_items):
    api_key = os.getenv("OPENAI_API_KEY")

    if not (OPENAI_OK and api_key):
        return (
            "GENAI CORE: OFFLINE — no API key configured. "
            "The diagnosis above is generated by the ML model and SHAP explanation."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an engineering assistant for a centrifugal-pump condition-monitoring system.

Do not invent readings and do not claim synthetic/demo data is experimental.

Fault: {fault}
Confidence: {confidence:.1%}
Prototype health score: {score:.1f}/100
Features: {features}
Top SHAP contributions: {shap_items}

Explain:
1. What the prediction means.
2. Why the model likely predicted it.
3. What an engineer should inspect.
4. Why the result must be validated using experimental data.
"""

    response = client.responses.create(
        model=model_name,
        input=prompt
    )
    return response.output_text


# ============================================================
# STATE
# ============================================================
init_db()

defaults = {
    "data": None,
    "models": None,
    "model_results": None,
    "best_model_name": None,
    "selected_features": None,
    "last_result": None,
    "raw_signal": None,
    "fft_data": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="padding:8px 0 16px 0;">
<div style="font-family:Orbitron,sans-serif;font-size:21px;font-weight:700;">
⚙ PUMPGUARD AI
</div>
<div style="color:#718b9e;font-size:10px;letter-spacing:1px;margin-top:5px;">
CONDITION MONITORING CORE
</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "SYSTEM NAVIGATION",
    [
        "Dashboard",
        "Signal Analysis",
        "ML Diagnosis",
        "Explainable AI",
        "Maintenance",
        "History"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**DATA INPUT**")

fs = st.sidebar.number_input(
    "Sampling frequency (Hz)",
    min_value=100.0,
    max_value=100000.0,
    value=2000.0,
    step=100.0
)

uploaded = st.sidebar.file_uploader(
    "Upload pump CSV",
    type=["csv"]
)

if st.sidebar.button("LOAD DEMO DATA"):
    st.session_state.data = make_demo_dataset()
    st.session_state.raw_signal = None
    st.session_state.fft_data = None
    st.session_state.last_result = None
    st.sidebar.success("Demo dataset loaded.")

if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.session_state.data = df_upload

        found = standardize_columns(df_upload)

        if found["vibration"] is not None:
            signal = clean_numeric_signal(df_upload[found["vibration"]])
            st.session_state.raw_signal = signal
            st.session_state.fft_data = fft_features(signal, fs)

        st.sidebar.success(
            f"{uploaded.name} // {len(df_upload):,} rows loaded"
        )
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")

st.sidebar.divider()

st.sidebar.markdown("**AI CORE STATUS**")
for name, state, cls in [
    ("DATA ENGINE", "ONLINE", ""),
    ("FFT ENGINE", "ONLINE", ""),
    ("ML ENGINE", "ONLINE", ""),
    ("XAI / SHAP", "ONLINE" if SHAP_OK else "OFFLINE", ""),
    ("DATABASE", "ONLINE", ""),
    ("RASPBERRY PI", "STANDBY", "off")
]:
    dot = "status-dot off" if state == "STANDBY" else "status-dot"
    st.sidebar.markdown(
        f'<div class="core-row"><span class="core-name">{name}</span>'
        f'<span class="core-state"><span class="{dot}"></span>{state}</span></div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    '<div class="small-note" style="margin-top:14px;">'
    'Prototype mode // experimental sensor validation pending'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP BAR
# ============================================================
st.markdown("""
<div class="topbar">
  <div class="brand">⚙ PUMPGUARD AI <span style='font-size:10px;color:#00d8ff;'>V3 // MULTI-MODEL</span></div>
  <div class="subtitle">
    CENTRIFUGAL PUMP // DIGITAL TWIN // FAULT INTELLIGENCE // PREDICTIVE MAINTENANCE
  </div>
  <div style="margin-top:10px;">
    <span class="tag">AI CORE ONLINE</span>
    <span class="tag">10 ML MODELS</span>
    <span class="tag">ENSEMBLE READY</span>
    <span class="tag">XAI READY</span>
    <span class="tag">CSV INPUT</span>
    <span class="tag">RASPBERRY PI READY</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NO DATA
# ============================================================
if st.session_state.data is None:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">SYSTEM INITIALIZATION</div>
      <div style="font-size:16px;color:#d8edf5;">
        Upload a labelled pump CSV or press <b>LOAD DEMO DATA</b>.
      </div>
      <div class="small-note" style="margin-top:10px;">
        Planned flow:
        CSV / Sensors → Cleaning → FFT → Feature Extraction →
        Sensor Fusion → ML → SHAP/XAI → Health Score →
        Maintenance Recommendation → History
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.data.copy()


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":

    feature_table, err = prepare_training_table(df, fs)

    if err:
        st.error(err)
        st.dataframe(df.head(20), use_container_width=True)
        st.stop()

    feats = available_features(feature_table)

    if "Fault" not in feature_table.columns or len(feats) < 2:
        st.warning(
            "A labelled dataset with at least two numeric features is required."
        )
        st.dataframe(feature_table.head(20), use_container_width=True)
        st.stop()

    try:
        fitted, results, best_name, X_train, X_test, y_train, y_test = train_models(
            feature_table, feats
        )

        st.session_state.models = fitted
        st.session_state.model_results = results
        st.session_state.best_model_name = best_name
        st.session_state.selected_features = feats

        model = fitted[best_name]

        # The dashboard diagnosis is deliberately the latest row.
        # Model performance itself is reported separately on ML Diagnosis.
        latest = feature_table.iloc[[-1]][feats].copy()
        latest = latest.fillna(feature_table[feats].median()).fillna(0)

        pred = model.predict(latest)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(latest)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 0.0

        feature_dict = feature_table.iloc[-1].to_dict()
        score = health_score(str(pred), confidence, feature_dict)

        result = {
            "fault": str(pred),
            "confidence": confidence,
            "health_score": score,
            **{
                k: feature_dict.get(k, np.nan)
                for k in BASE_FEATURES
            },
            "source": "CSV / Prototype"
        }

        st.session_state.last_result = result

        status = health_label(score)

        # ---- Hero row ----
        left, mid, right = st.columns([1.0, 1.15, 1.25])

        with left:
            st.markdown(pump_svg("RUNNING"), unsafe_allow_html=True)

        with mid:
            deg = int(score * 3.6)
            st.markdown(
                f"""
                <div class="panel" style="text-align:center;">
                  <div class="panel-title">SYSTEM HEALTH INDEX</div>
                  <div class="health-ring" style="--scoredeg:{deg}deg;">
                    <div class="health-inner">
                      <div class="health-value">{score:.0f}</div>
                      <div class="health-label">/ 100</div>
                    </div>
                  </div>
                  <div style="font-family:Orbitron,sans-serif;color:#bfeeff;">
                    {status}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                f"""
                <div class="fault-card">
                  <div class="panel-title">AI DIAGNOSTIC ENGINE</div>
                  <div style="color:#7f96a7;font-size:11px;">PRIMARY PREDICTION</div>
                  <div class="fault-name">⚠ {pred}</div>
                  <div class="confidence">
                    MODEL: <b>{best_name.upper()}</b>
                    &nbsp; // &nbsp;
                    CONFIDENCE: <b>{confidence:.1%}</b>
                  </div>
                  <div style="margin-top:15px;">
                    <span class="tag">FAULT CLASSIFIER</span>
                    <span class="tag">XAI READY</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---- Sensor row ----
        st.markdown('<div class="panel-title">LIVE SENSOR / FEATURE SNAPSHOT</div>',
                    unsafe_allow_html=True)

        sensor_items = [
            ("VIBRATION RMS", feature_dict.get("RMS", np.nan), "", ".3f"),
            ("PEAK", feature_dict.get("Peak", np.nan), "", ".3f"),
            ("DOMINANT FREQ.", feature_dict.get("Dominant Frequency", np.nan), "Hz", ".1f"),
            ("TEMPERATURE", feature_dict.get("Temperature", np.nan), "°C", ".1f"),
            ("MOTOR CURRENT", feature_dict.get("Current", np.nan), "A", ".2f"),
            ("ROTATIONAL SPEED", feature_dict.get("RPM", np.nan), "RPM", ".0f")
        ]

        cols = st.columns(6)
        for col, (name, value, unit, fmt) in zip(cols, sensor_items):
            if pd.isna(value):
                val = "N/A"
            else:
                val = format(float(value), fmt)
            with col:
                st.markdown(
                    f"""
                    <div class="sensor-card">
                      <div class="sensor-name">{name}</div>
                      <div class="sensor-value">{val}</div>
                      <div class="sensor-unit">{unit}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        # ---- Recommendation ----
        a, b = st.columns([1.2, 1])

        with a:
            st.markdown(
                f"""
                <div class="panel">
                  <div class="panel-title">MAINTENANCE INTELLIGENCE</div>
                  <div class="ai-box">
                    <b>RECOMMENDED ACTION</b><br><br>
                    {recommendation(str(pred))}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with b:
            st.markdown(
                f"""
                <div class="panel">
                  <div class="panel-title">AI PIPELINE STATUS</div>
                  <div class="core-row"><span class="core-name">INPUT</span><span class="core-state">CSV</span></div>
                  <div class="core-row"><span class="core-name">FEATURES</span><span class="core-state">{len(feats)} ACTIVE</span></div>
                  <div class="core-row"><span class="core-name">MODEL</span><span class="core-state">{best_name}</span></div>
                  <div class="core-row"><span class="core-name">XAI</span><span class="core-state">SHAP READY</span></div>
                  <div class="core-row"><span class="core-name">DATABASE</span><span class="core-state">READY</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("SAVE DIAGNOSIS TO HISTORY"):
            save_history(result)
            st.success("Diagnosis stored in PumpGuard history.")

        st.caption(
            "Prototype note: health score and maintenance rules are not experimentally calibrated yet."
        )

    except Exception as e:
        st.error(f"Dashboard error: {e}")


# ============================================================
# SIGNAL ANALYSIS
# ============================================================
elif page == "Signal Analysis":

    st.subheader("Signal Intelligence")

    found = standardize_columns(df)
    raw_col = found["vibration"]

    if raw_col is None:
        st.info(
            "This CSV contains extracted features rather than a raw vibration waveform."
        )
        feature_table, err = prepare_training_table(df, fs)

        if not err:
            st.markdown(
                '<div class="panel"><div class="panel-title">EXTRACTED FEATURE MATRIX</div>',
                unsafe_allow_html=True
            )
            st.dataframe(feature_table.head(50), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        signal = clean_numeric_signal(df[raw_col])
        freqs, spectrum, feats = fft_features(signal, fs)

        if freqs is None:
            st.error("Not enough samples for FFT.")
            st.stop()

        st.session_state.raw_signal = signal
        st.session_state.fft_data = (freqs, spectrum, feats)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<div class="panel"><div class="panel-title">TIME DOMAIN // VIBRATION</div>',
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            t = np.arange(len(signal)) / fs
            ax.plot(t, signal)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=.2)
            st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                '<div class="panel"><div class="panel-title">FFT // FREQUENCY DOMAIN</div>',
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            mask = freqs <= 400
            ax.plot(freqs[mask], spectrum[mask])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.grid(alpha=.2)
            st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="panel"><div class="panel-title">FEATURE EXTRACTION ENGINE</div>',
            unsafe_allow_html=True
        )
        st.dataframe(
            pd.DataFrame([feats]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ML DIAGNOSIS
# ============================================================
elif page == "ML Diagnosis":

    st.subheader("Machine Learning Fault Diagnosis")

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">AI MODEL SUITE</div>
          <div class="small-note">
            Tree ensembles: Random Forest, Extra Trees, Gradient Boosting,
            Histogram Gradient Boosting, AdaBoost •
            Classical ML: Decision Tree, SVM, KNN, Logistic Regression •
            Neural ML: MLP Neural Network •
            Ensemble: Soft Voting
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    feature_table, err = prepare_training_table(df, fs)

    if err:
        st.error(err)
        st.stop()

    feats = available_features(feature_table)

    selected = st.multiselect(
        "FEATURE VECTOR",
        feats,
        default=feats
    )

    if len(selected) < 2:
        st.warning("Select at least two features.")
        st.stop()

    if st.button("TRAIN / COMPARE AI MODELS", type="primary"):
        try:
            with st.spinner("Training models and evaluating test set..."):
                fitted, results, best_name, X_train, X_test, y_train, y_test = train_models(
                    feature_table, selected
                )

                st.session_state.models = fitted
                st.session_state.model_results = results
                st.session_state.best_model_name = best_name
                st.session_state.selected_features = selected

            st.success(f"Best prototype model: {best_name}")

        except Exception as e:
            st.error(f"Training error: {e}")

    if st.session_state.model_results:

        results = st.session_state.model_results

        score_rows = []
        for name, r in results.items():
            score_rows.append({
                "MODEL": name,
                "TEST ACCURACY": r["accuracy"],
                "PRECISION": r["precision"],
                "RECALL": r["recall"],
                "F1 SCORE": r["f1"],
                "5-FOLD CV": r["cv_mean"],
                "CV STD": r["cv_std"]
            })

        score_df = pd.DataFrame(score_rows).sort_values(
            ["F1 SCORE", "5-FOLD CV"], ascending=False
        )
        score_df.insert(0, "RANK", range(1, len(score_df) + 1))

        st.markdown(
            '<div class="panel"><div class="panel-title">MULTI-MODEL BENCHMARK // TEST + CROSS-VALIDATION</div>',
            unsafe_allow_html=True
        )
        st.dataframe(
            score_df.style.format({
                "TEST ACCURACY": "{:.2%}",
                "PRECISION": "{:.2%}",
                "RECALL": "{:.2%}",
                "F1 SCORE": "{:.2%}",
                "5-FOLD CV": "{:.2%}",
                "CV STD": "{:.2%}"
            }),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="ai-box">
              <b>MODEL SELECTION:</b> {best_name}<br>
              Selection priority: weighted F1 → mean 5-fold CV accuracy → test accuracy.
              The held-out test set is kept separate from cross-validation.
            </div>
            """,
            unsafe_allow_html=True
        )

        best_name = st.session_state.best_model_name
        st.markdown(
            f"""
            <div class="ai-box">
              <b>SELECTED MODEL:</b> {best_name}<br>
              Model selection is based on weighted F1 score, then accuracy.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Classification Report")
        report = results[best_name]["report"]
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

        st.markdown("### Confusion Matrix")
        cm = results[best_name]["cm"]
        labels = results[best_name]["labels"]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)

        st.warning(
            "If this dataset is synthetic or artificially well separated, "
            "very high scores are only a software-pipeline result. "
            "Final project performance must be reported using experimental pump data. "
            "Use the CV mean and CV standard deviation together with the held-out test metrics."

        )


# ============================================================
# EXPLAINABLE AI
# ============================================================
elif page == "Explainable AI":

    st.subheader("Explainable AI // SHAP")

    if not SHAP_OK:
        st.error("SHAP is not installed.")
        st.stop()

    if st.session_state.models is None:
        st.info("Train the ML models first.")
        st.stop()

    model_names = list(st.session_state.models.keys())

    default_index = (
        model_names.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in model_names
        else 0
    )

    model_name = st.selectbox(
        "MODEL TO EXPLAIN",
        model_names,
        index=default_index
    )

    model = st.session_state.models[model_name]

    feature_table, err = prepare_training_table(df, fs)

    if err:
        st.error(err)
        st.stop()

    feats = st.session_state.selected_features or available_features(feature_table)

    X = feature_table[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    row_number = st.number_input(
        "ROW TO EXPLAIN",
        min_value=0,
        max_value=max(0, len(X) - 1),
        value=max(0, len(X) - 1),
        step=1
    )

    x_row = X.iloc[[int(row_number)]]
    pred = model.predict(x_row)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_row)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 0.0

    if model_name not in [
        "Random Forest", "Extra Trees", "Gradient Boosting",
        "Decision Tree", "AdaBoost"
    ]:
        st.info(
            "Tree-based SHAP is enabled for Random Forest, Extra Trees, "
            "Gradient Boosting, Decision Tree and AdaBoost. "
            "Select one of these models for a direct SHAP explanation."
        )
        st.stop()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_row)

        classes = list(model.classes_)
        class_index = classes.index(pred)

        if isinstance(shap_values, list):
            vals = np.asarray(shap_values[class_index])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                vals = arr[0, :, class_index]
            else:
                vals = arr[0]

        shap_df = pd.DataFrame({
            "FEATURE": feats,
            "SHAP": vals,
            "ABS SHAP": np.abs(vals),
            "VALUE": x_row.iloc[0].values
        }).sort_values("ABS SHAP", ascending=False)

        st.markdown(
            f"""
            <div class="fault-card">
              <div class="panel-title">XAI DIAGNOSTIC EXPLANATION</div>
              <div class="fault-name">{pred}</div>
              <div class="confidence">
                MODEL CONFIDENCE // {confidence:.1%}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1.25, 1])

        with c1:
            st.markdown(
                '<div class="panel"><div class="panel-title">FEATURE CONTRIBUTION</div>',
                unsafe_allow_html=True
            )

            chart_df = shap_df.head(10).set_index("FEATURE")["SHAP"]
            st.bar_chart(chart_df)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                '<div class="panel"><div class="panel-title">ENGINEERING EVIDENCE</div>',
                unsafe_allow_html=True
            )

            for row in shap_df.head(6).itertuples():
                direction = "toward prediction" if row.SHAP >= 0 else "against prediction"
                st.markdown(
                    f"""
                    <div class="core-row">
                      <span class="core-name">{row.FEATURE}</span>
                      <span class="core-state">{row.SHAP:+.3f}</span>
                    </div>
                    <div class="small-note">
                      measured value: {row.VALUE:.4g} // {direction}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="ai-box">
            <b>ENGINEERING INTERPRETATION</b><br>
            SHAP identifies which input features contributed most to the model output.
            A SHAP contribution explains the model's decision; it does not by itself
            prove the physical root cause of the machine fault.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Optional GenAI Explanation")

        if st.button("GENERATE ENGINEER-FRIENDLY EXPLANATION"):
            top_items = [
                f"{r.FEATURE}={r.VALUE:.4g} (SHAP {r.SHAP:+.3f})"
                for r in shap_df.head(5).itertuples()
            ]

            feature_dict = x_row.iloc[0].to_dict()
            score = health_score(str(pred), confidence, feature_dict)

            with st.spinner("Generating engineering explanation..."):
                output = genai_explanation(
                    str(pred),
                    confidence,
                    score,
                    feature_dict,
                    top_items
                )

            st.markdown(
                f'<div class="panel"><div class="panel-title">GENAI ENGINEERING REPORT</div>{output}</div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"SHAP error: {e}")


# ============================================================
# MAINTENANCE
# ============================================================
elif page == "Maintenance":

    st.subheader("Maintenance Intelligence")

    result = st.session_state.last_result

    if result is None:
        st.info("Generate a diagnosis from the Dashboard first.")
        st.stop()

    fault = result["fault"]
    score = result["health_score"]

    st.markdown(
        f"""
        <div class="fault-card">
          <div class="panel-title">CURRENT DIAGNOSIS</div>
          <div class="fault-name">{fault}</div>
          <div class="confidence">HEALTH INDEX // {score:.1f}/100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Recommended inspection")

    st.markdown(
        f'<div class="ai-box">{recommendation(fault)}</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Engineering workflow")

    if fault == "Misalignment":
        steps = [
            "Inspect coupling condition.",
            "Check motor-pump shaft alignment.",
            "Inspect foundation and mounting.",
            "Repeat vibration measurement after correction."
        ]
    elif fault == "Impeller Damage":
        steps = [
            "Inspect impeller vanes.",
            "Check for blockage, wear or mechanical damage.",
            "Check rotor / impeller balance.",
            "Repeat vibration spectrum measurement."
        ]
    else:
        steps = [
            "Continue monitoring.",
            "Compare current readings with historical baseline.",
            "Validate diagnosis using experimental inspection."
        ]

    for i, step in enumerate(steps, 1):
        st.markdown(
            f"""
            <div class="core-row">
              <span class="core-name">STEP {i:02d}</span>
              <span style="color:#cfe9f2;">{step}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "Prototype decision-support output. Final maintenance actions must be "
        "validated by qualified engineering inspection."
    )


# ============================================================
# HISTORY
# ============================================================
elif page == "History":

    st.subheader("Machine History // Database")

    hist = load_history()

    if hist.empty:
        st.info("No saved diagnoses yet.")
    else:
        st.dataframe(hist, use_container_width=True)

        trend = hist.copy()
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        trend = trend.sort_values("timestamp").set_index("timestamp")

        st.markdown(
            '<div class="panel"><div class="panel-title">HEALTH TREND</div>',
            unsafe_allow_html=True
        )
        st.line_chart(trend[["health_score"]])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "DOWNLOAD HISTORY CSV",
            hist.to_csv(index=False).encode("utf-8"),
            file_name="pump_diagnosis_history.csv",
            mime="text/csv"
        )
