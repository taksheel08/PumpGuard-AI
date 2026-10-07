
import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="PumpGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.block-container {padding-top:1.2rem;}
.metric-card {
    padding: 18px; border-radius: 14px; background: white;
    border: 1px solid #e7ebf0; box-shadow: 0 2px 10px rgba(0,0,0,.04);
}
.small {font-size:0.85rem;color:#6b7280;}
.fault {padding:14px;border-radius:12px;background:#fff4f4;border:1px solid #ffd1d1;}
.good {padding:14px;border-radius:12px;background:#f0fff5;border:1px solid #c7f0d5;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================
DB_FILE = "pump_history.db"

def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
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
        result["fault"],
        result["confidence"],
        result["health_score"],
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
    df = pd.read_sql_query("SELECT * FROM diagnosis_history ORDER BY id DESC", con)
    con.close()
    return df


# ============================================================
# SIGNAL PROCESSING
# ============================================================
def clean_numeric_signal(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values.astype(float)
    if len(x) == 0:
        return np.array([0.0])
    # remove NaN/inf and simple DC component
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    return x

def fft_features(signal, fs):
    signal = clean_numeric_signal(signal)
    n = len(signal)
    if n < 8:
        return None, None, {}

    window = np.hanning(n)
    y = signal * window
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # Ignore DC component
    if len(spectrum) > 1:
        idx = np.argmax(spectrum[1:]) + 1
    else:
        idx = 0

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
# COLUMN / DATA HANDLING
# ============================================================
ALIASES = {
    "vibration": ["vibration", "acceleration", "amplitude", "signal", "vibration_signal", "accel"],
    "temperature": ["temperature", "temp", "temperature_c", "temp_c"],
    "current": ["current", "motor_current", "current_a", "amps", "ampere"],
    "rpm": ["rpm", "speed", "motor_speed"],
    "label": ["fault", "fault_type", "label", "class", "condition", "status"]
}

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
    out = df.copy()
    found = {}
    for key, aliases in ALIASES.items():
        found[key] = find_column(out, aliases)
    return found

def make_demo_dataset(n_per_class=160, seed=42):
    """
    Synthetic demonstration dataset only.
    It is NOT experimental pump data and must not be reported as measured data.
    """
    rng = np.random.default_rng(seed)
    classes = ["Healthy", "Misalignment", "Impeller Damage"]
    rows = []

    specs = {
        "Healthy": {
            "rms": (0.18, 0.04), "peak": (0.55, 0.10), "p2p": (1.05, 0.18),
            "std": (0.17, 0.035), "kurt": (2.7, 0.35), "skew": (0.02, 0.12),
            "crest": (3.0, 0.35), "freq": (50, 3), "temp": (34, 2),
            "current": (2.2, 0.25), "rpm": (2910, 12)
        },
        "Misalignment": {
            "rms": (0.58, 0.10), "peak": (1.55, 0.25), "p2p": (3.0, 0.40),
            "std": (0.52, 0.08), "kurt": (3.8, 0.55), "skew": (0.10, 0.18),
            "crest": (3.1, 0.40), "freq": (100, 5), "temp": (42, 3),
            "current": (3.1, 0.35), "rpm": (2895, 18)
        },
        "Impeller Damage": {
            "rms": (0.82, 0.14), "peak": (2.25, 0.35), "p2p": (4.4, 0.55),
            "std": (0.75, 0.12), "kurt": (5.0, 0.8), "skew": (0.25, 0.22),
            "crest": (3.35, 0.45), "freq": (298, 7), "temp": (47, 3),
            "current": (3.5, 0.40), "rpm": (2888, 22)
        }
    }

    for cls in classes:
        s = specs[cls]
        for _ in range(n_per_class):
            rows.append({
                "RMS": max(0, rng.normal(*s["rms"])),
                "Peak": max(0, rng.normal(*s["peak"])),
                "Peak-to-Peak": max(0, rng.normal(*s["p2p"])),
                "Std": max(0, rng.normal(*s["std"])),
                "Kurtosis": max(0.1, rng.normal(*s["kurt"])),
                "Skewness": rng.normal(*s["skew"]),
                "Crest Factor": max(0.5, rng.normal(*s["crest"])),
                "Dominant Frequency": max(1, rng.normal(*s["freq"])),
                "Temperature": rng.normal(*s["temp"]),
                "Current": max(0.1, rng.normal(*s["current"])),
                "RPM": max(2500, rng.normal(*s["rpm"])),
                "Fault": cls
            })
    return pd.DataFrame(rows)

def prepare_training_table(df, fs):
    """
    Supports two common inputs:
    1) Feature CSV: one row/sample with feature columns + Fault/Label.
    2) Raw vibration CSV: one vibration column + optional sensors + Fault/Label.
       For raw data, a global FFT/statistical feature vector is produced.
    """
    found = standardize_columns(df)
    label_col = found["label"]

    if label_col is None:
        return None, "No fault/label column found. Add a column such as Fault or Label."

    raw_col = found["vibration"]
    feature_names = [
        "RMS", "Peak", "Peak-to-Peak", "Std", "Kurtosis",
        "Skewness", "Crest Factor", "Dominant Frequency"
    ]

    # Already-featured dataset
    existing = {f: find_column(df, [f.lower(), f.lower().replace("-", "_"), f.lower().replace(" ", "_")])
                for f in feature_names}

    if sum(v is not None for v in existing.values()) >= 5:
        out = pd.DataFrame()
        for f, c in existing.items():
            if c is not None:
                out[f] = pd.to_numeric(df[c], errors="coerce")
        for sensor_key, col in [("Temperature", found["temperature"]),
                                ("Current", found["current"]),
                                ("RPM", found["rpm"])]:
            if col is not None:
                out[sensor_key] = pd.to_numeric(df[col], errors="coerce")
        out["Fault"] = df[label_col].astype(str).values
        out = out.dropna()
        return out, None

    # Raw signal dataset
    if raw_col is None:
        return None, "Could not find enough feature columns and no raw vibration column was found."

    signal = clean_numeric_signal(df[raw_col])
    freqs, spec, feats = fft_features(signal, fs)
    if not feats:
        return None, "The vibration signal is too short for FFT feature extraction."

    # Optional sensor values: use median of uploaded segment
    feats["Temperature"] = float(pd.to_numeric(df[found["temperature"]], errors="coerce").median()) if found["temperature"] else np.nan
    feats["Current"] = float(pd.to_numeric(df[found["current"]], errors="coerce").median()) if found["current"] else np.nan
    feats["RPM"] = float(pd.to_numeric(df[found["rpm"]], errors="coerce").median()) if found["rpm"] else np.nan

    # If raw signal has labels per sample, use the most common label for the segment.
    label = df[label_col].dropna().astype(str).mode()
    label = label.iloc[0] if len(label) else "Unknown"

    return pd.DataFrame([{**feats, "Fault": label}]), None


# ============================================================
# ML
# ============================================================
BASE_FEATURES = [
    "RMS", "Peak", "Peak-to-Peak", "Std", "Kurtosis",
    "Skewness", "Crest Factor", "Dominant Frequency",
    "Temperature", "Current", "RPM"
]

def available_features(df):
    return [c for c in BASE_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def train_models(df, selected_features):
    X = df[selected_features].copy()
    y = df["Fault"].astype(str)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    if y.nunique() < 2:
        raise ValueError("At least two fault classes are required for classification.")

    # Stratified split where possible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced", random_state=42))
        ])
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = {
            "accuracy": acc,
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "cm": confusion_matrix(y_test, pred, labels=sorted(y.unique())),
            "labels": sorted(y.unique())
        }
        fitted[name] = model

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    return fitted, results, best_name, X_train, X_test, y_train, y_test


# ============================================================
# HEALTH SCORE + RECOMMENDATIONS
# ============================================================
def health_score(fault, confidence, features):
    # Prototype index for demonstration until experimental health-score labels exist.
    base = {"Healthy": 95, "Good": 85, "Misalignment": 62, "Impeller Damage": 50}
    if fault in base:
        score = base[fault]
    else:
        score = max(20, 85 - 50 * confidence)

    # Sensor stress adjustment
    if features.get("Temperature", np.nan) > 50:
        score -= 8
    if features.get("Current", np.nan) > 4:
        score -= 7
    if features.get("RMS", np.nan) > 1.2:
        score -= 8

    score = float(np.clip(score + 5 * (confidence - 0.5), 0, 100))
    return score

def health_label(score):
    if score >= 80:
        return "Healthy"
    if score >= 60:
        return "Good / Monitor"
    if score >= 40:
        return "Service Required"
    return "Critical"

def recommendation(fault):
    recs = {
        "Healthy": "Continue normal operation. Keep monitoring vibration and sensor trends.",
        "Misalignment": "Inspect coupling and shaft alignment. Check for 1×/2× rotational components and realign the motor-pump shafts if required.",
        "Impeller Damage": "Inspect the impeller for vane damage, imbalance or blockage. Verify the vibration spectrum and pump operating condition.",
        "Bearing Fault": "Inspect bearing condition, lubrication and mounting. Confirm the characteristic bearing-frequency components.",
        "Rotor Unbalance": "Inspect rotor/impeller balance and check for material buildup or damage.",
        "Loose Foundation": "Inspect foundation bolts, baseplate and mounting stiffness.",
        "Unknown": "Collect more representative labelled data before taking maintenance action."
    }
    return recs.get(fault, "Inspect the pump and verify the diagnosis with additional measurements.")


# ============================================================
# OPTIONAL GENAI
# ============================================================
def genai_explanation(fault, confidence, score, features, shap_items):
    api_key = os.getenv("OPENAI_API_KEY")
    if not (OPENAI_OK and api_key):
        return (
            "GenAI is not connected yet. The app is using the ML + SHAP result directly. "
            "Set OPENAI_API_KEY to enable natural-language maintenance explanations."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an engineering assistant for a centrifugal-pump condition-monitoring project.
Do not invent sensor readings or claim that synthetic data is experimental.
Explain the following ML diagnosis in simple engineering language.

Fault: {fault}
Confidence: {confidence:.1%}
Prototype health score: {score:.1f}/100
Features: {features}
Top SHAP contributions: {shap_items}

Give:
1. What the diagnosis means.
2. Why the model likely predicted it.
3. What an engineer should inspect.
4. A short caution that the result must be validated using experimental data.
"""
    response = client.responses.create(model=model_name, input=prompt)
    return response.output_text


# ============================================================
# SESSION STATE
# ============================================================
init_db()

if "data" not in st.session_state:
    st.session_state.data = None
if "models" not in st.session_state:
    st.session_state.models = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "raw_signal" not in st.session_state:
    st.session_state.raw_signal = None
if "fft_data" not in st.session_state:
    st.session_state.fft_data = None


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ PumpGuard AI")
st.sidebar.caption("Smart centrifugal-pump fault identification")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Signal Analysis", "ML Diagnosis", "Explainable AI", "Maintenance", "History"]
)

st.sidebar.divider()
st.sidebar.subheader("Data Input")

fs = st.sidebar.number_input(
    "Sampling frequency (Hz)",
    min_value=100.0, max_value=100000.0, value=2000.0, step=100.0,
    help="Used only when the uploaded file contains a raw vibration/time signal."
)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("Use Demo Dataset"):
    st.session_state.data = make_demo_dataset()
    st.session_state.raw_signal = None
    st.session_state.fft_data = None
    st.session_state.last_result = None
    st.sidebar.success("Synthetic demo data loaded.")

if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.session_state.data = df_upload
        st.session_state.raw_signal = None
        st.session_state.fft_data = None

        found = standardize_columns(df_upload)
        if found["vibration"] is not None:
            st.session_state.raw_signal = clean_numeric_signal(df_upload[found["vibration"]])
            st.session_state.fft_data = fft_features(st.session_state.raw_signal, fs)

        st.sidebar.success(f"Loaded {len(df_upload):,} rows.")
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")

st.sidebar.divider()
st.sidebar.caption("Prototype note: current app can use synthetic/demo data or labelled CSVs. Final diagnosis must be validated using your experimental pump readings.")


# ============================================================
# HEADER
# ============================================================
st.title("⚙️ PumpGuard AI")
st.caption("Smart Digital Twin-Based Fault Identification & Predictive Maintenance — prototype dashboard")

if st.session_state.data is None:
    st.info("Start with **Use Demo Dataset** to test the complete workflow, then replace it with your labelled CSV readings.")
    st.markdown("""
    ### Planned data flow
    **CSV / sensors → cleaning → FFT → feature extraction → sensor fusion → ML → SHAP/XAI → health score → maintenance recommendation → history/dashboard**
    """)
    st.stop()

df = st.session_state.data.copy()


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    st.subheader("Live Machine Health Dashboard")

    features_df, prep_error = prepare_training_table(df, fs)

    if prep_error:
        st.warning(prep_error)
        st.write("Detected columns:", list(df.columns))
        st.stop()

    # Train automatically when possible
    feats = available_features(features_df)
    if "Fault" not in features_df.columns or len(feats) < 2:
        st.warning("The dashboard needs labelled data and at least two numeric features for ML diagnosis.")
        st.dataframe(features_df.head(20), use_container_width=True)
        st.stop()

    try:
        fitted, results, best_name, X_train, X_test, y_train, y_test = train_models(features_df, feats)
        st.session_state.models = fitted
        st.session_state.model_results = results
        st.session_state.best_model_name = best_name
        st.session_state.selected_features = feats

        model = fitted[best_name]
        latest = features_df.iloc[[-1]][feats].copy()
        latest = latest.fillna(features_df[feats].median()).fillna(0)
        pred = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0] if hasattr(model, "predict_proba") else None
        classes = list(model.classes_)
        confidence = float(np.max(proba)) if proba is not None else 0.0

        feature_dict = features_df.iloc[-1].to_dict()
        score = health_score(pred, confidence, feature_dict)

        result = {
            "fault": str(pred),
            "confidence": confidence,
            "health_score": score,
            **{k: feature_dict.get(k, np.nan) for k in BASE_FEATURES},
            "source": "Demo/CSV"
        }
        st.session_state.last_result = result

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machine Status", health_label(score))
        c2.metric("Health Score", f"{score:.1f}/100")
        c3.metric("Detected Fault", str(pred))
        c4.metric("Confidence", f"{confidence:.1%}")

        st.divider()

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Sensor / Feature Snapshot")
            snap_cols = [c for c in ["RMS", "Peak", "Peak-to-Peak", "Dominant Frequency",
                                     "Temperature", "Current", "RPM"] if c in features_df.columns]
            st.dataframe(features_df[snap_cols].tail(10), use_container_width=True)

        with right:
            st.subheader("Maintenance Recommendation")
            if pred == "Healthy":
                st.markdown(f'<div class="good">✅ {recommendation(pred)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fault">⚠️ {recommendation(pred)}</div>', unsafe_allow_html=True)

        if st.button("Save diagnosis to history"):
            save_history(result)
            st.success("Diagnosis saved to SQLite history.")

        st.subheader("Recent Trend")
        history = load_history()
        if not history.empty:
            chart_cols = [c for c in ["health_score", "confidence"] if c in history.columns]
            st.line_chart(history.sort_values("timestamp")[chart_cols].set_index(
                pd.to_datetime(history.sort_values("timestamp")["timestamp"])
            ))
        else:
            st.info("No saved diagnoses yet.")

    except Exception as e:
        st.error(f"Dashboard diagnosis error: {e}")


# ============================================================
# SIGNAL ANALYSIS
# ============================================================
elif page == "Signal Analysis":
    st.subheader("Signal Processing & Feature Extraction")

    found = standardize_columns(df)
    raw_col = found["vibration"]

    if raw_col is None:
        st.info("No raw vibration column detected. Your uploaded file appears to contain extracted features.")
        feature_table, err = prepare_training_table(df, fs)
        if err:
            st.error(err)
        else:
            st.subheader("Available extracted features")
            st.dataframe(feature_table.head(50), use_container_width=True)
    else:
        signal = clean_numeric_signal(df[raw_col])
        freqs, spectrum, feats = fft_features(signal, fs)

        if freqs is None:
            st.error("Not enough samples for FFT.")
        else:
            st.session_state.raw_signal = signal
            st.session_state.fft_data = (freqs, spectrum, feats)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Time-domain vibration")
                fig, ax = plt.subplots(figsize=(8, 4))
                t = np.arange(len(signal)) / fs
                ax.plot(t, signal)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.grid(alpha=0.25)
                st.pyplot(fig, clear_figure=True)

            with c2:
                st.markdown("#### Frequency-domain spectrum")
                fig, ax = plt.subplots(figsize=(8, 4))
                mask = freqs <= 400
                ax.plot(freqs[mask], spectrum[mask])
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.grid(alpha=0.25)
                st.pyplot(fig, clear_figure=True)

            st.subheader("Extracted Features")
            feature_df = pd.DataFrame([feats]).T.rename(columns={0: "Value"})
            st.dataframe(feature_df, use_container_width=True)

            st.caption("The project workflow calls for FFT-based frequency-domain processing and statistical feature extraction. Your progress presentation also shows a healthy spectrum around 50 Hz and a faulty example around 298 Hz; these are useful reference observations, not universal thresholds. ")


# ============================================================
# ML DIAGNOSIS
# ============================================================
elif page == "ML Diagnosis":
    st.subheader("Machine Learning Fault Diagnosis")

    feature_table, err = prepare_training_table(df, fs)
    if err:
        st.error(err)
        st.stop()

    feats = available_features(feature_table)
    if "Fault" not in feature_table.columns or len(feats) < 2:
        st.error("Need a Fault/Label column and at least two numeric features.")
        st.stop()

    default_feats = feats
    selected = st.multiselect(
        "Features used by ML",
        feats,
        default=default_feats
    )

    if len(selected) < 2:
        st.warning("Select at least two features.")
        st.stop()

    if st.button("Train / Compare ML Models", type="primary"):
        with st.spinner("Training Decision Tree, Random Forest and SVM..."):
            try:
                fitted, results, best_name, X_train, X_test, y_train, y_test = train_models(feature_table, selected)
                st.session_state.models = fitted
                st.session_state.model_results = results
                st.session_state.best_model_name = best_name
                st.session_state.selected_features = selected
                st.success(f"Best prototype model: {best_name}")
            except Exception as e:
                st.error(f"Training error: {e}")

    if st.session_state.model_results:
        rows = []
        for name, r in st.session_state.model_results.items():
            rows.append({"Model": name, "Test Accuracy": r["accuracy"]})
        score_df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
        st.dataframe(score_df, use_container_width=True)
        st.bar_chart(score_df.set_index("Model"))

        best = st.session_state.best_model_name
        st.write(f"**Selected model:** {best}")
        st.write("Classification report")
        report = st.session_state.model_results[best]["report"]
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

        st.caption("Do not report demo-data accuracy as experimental accuracy. Once you collect real healthy/faulty pump readings, retrain and report cross-validation/test results from that dataset.")


# ============================================================
# EXPLAINABLE AI
# ============================================================
elif page == "Explainable AI":
    st.subheader("🔎 Explainable AI — SHAP")

    if not SHAP_OK:
        st.error("SHAP is not installed. Run: pip install shap")
        st.stop()

    if st.session_state.models is None:
        st.info("Train the ML models first from the ML Diagnosis page.")
        st.stop()

    model_name = st.selectbox(
        "Model to explain",
        list(st.session_state.models.keys()),
        index=list(st.session_state.models.keys()).index(st.session_state.best_model_name)
    )
    model = st.session_state.models[model_name]
    feature_table, err = prepare_training_table(df, fs)

    if err:
        st.error(err)
        st.stop()

    feats = st.session_state.selected_features or available_features(feature_table)
    X = feature_table[feats].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if len(X) == 0:
        st.stop()

    row_number = st.number_input("Row to explain", min_value=0, max_value=max(0, len(X)-1), value=0, step=1)
    x_row = X.iloc[[int(row_number)]]

    pred = model.predict(x_row)[0]
    proba = model.predict_proba(x_row)[0] if hasattr(model, "predict_proba") else None
    confidence = float(np.max(proba)) if proba is not None else 0.0

    if model_name in ["Random Forest", "Decision Tree"]:
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
                "Feature": feats,
                "SHAP": vals,
                "Absolute SHAP": np.abs(vals),
                "Value": x_row.iloc[0].values
            }).sort_values("Absolute SHAP", ascending=False)

            st.metric("Prediction", str(pred), f"{confidence:.1%} confidence")
            st.bar_chart(shap_df.head(10).set_index("Feature")["SHAP"])
            st.dataframe(shap_df, use_container_width=True)

            top_items = [
                f"{r.Feature}={r.Value:.3f} (SHAP {r.SHAP:+.3f})"
                for r in shap_df.head(5).itertuples()
            ]

            st.markdown("### Engineering interpretation")
            st.write(
                f"The model predicts **{pred}**. The largest SHAP contributions show which "
                "input features pushed this prediction toward that class. SHAP explains the model output; "
                "it does not by itself prove a physical root cause."
            )

            st.markdown("### Optional GenAI explanation")
            if st.button("Generate engineer-friendly explanation"):
                feature_dict = x_row.iloc[0].to_dict()
                score = health_score(str(pred), confidence, feature_dict)
                with st.spinner("Generating explanation..."):
                    text_out = genai_explanation(str(pred), confidence, score, feature_dict, top_items)
                st.write(text_out)

        except Exception as e:
            st.error(f"SHAP explanation error: {e}")
    else:
        st.info("For the first prototype, use Random Forest or Decision Tree for SHAP because TreeExplainer is designed for tree-based models.")


# ============================================================
# MAINTENANCE
# ============================================================
elif page == "Maintenance":
    st.subheader("🔧 Maintenance Recommendation")

    result = st.session_state.last_result
    if result is None:
        st.info("Open Dashboard first to generate a diagnosis.")
        st.stop()

    fault = result["fault"]
    score = result["health_score"]

    st.metric("Current diagnosis", fault)
    st.metric("Prototype health score", f"{score:.1f}/100")
    st.write(recommendation(fault))

    if fault == "Misalignment":
        st.markdown("""
        **Suggested inspection sequence**
        1. Check coupling condition.
        2. Check motor-pump shaft alignment.
        3. Inspect base and mounting.
        4. Re-measure vibration after correction.
        """)
    elif fault == "Impeller Damage":
        st.markdown("""
        **Suggested inspection sequence**
        1. Inspect impeller vanes.
        2. Check for blockage, wear or damage.
        3. Check rotor/impeller balance.
        4. Re-measure the vibration spectrum.
        """)
    else:
        st.markdown("Continue monitoring and validate the result using the experimental setup.")

    st.warning("These are prototype decision-support recommendations, not a substitute for engineering inspection or safety procedures.")


# ============================================================
# HISTORY
# ============================================================
elif page == "History":
    st.subheader("📊 Diagnosis History")

    hist = load_history()
    if hist.empty:
        st.info("No records yet. Save a diagnosis from the Dashboard.")
    else:
        st.dataframe(hist, use_container_width=True)

        st.subheader("Health Score Trend")
        trend = hist.copy()
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        trend = trend.sort_values("timestamp").set_index("timestamp")
        st.line_chart(trend[["health_score"]])

        st.download_button(
            "Download history CSV",
            hist.to_csv(index=False).encode("utf-8"),
            file_name="pump_diagnosis_history.csv",
            mime="text/csv"
        )
