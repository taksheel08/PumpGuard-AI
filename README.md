
# PumpGuard AI — Centrifugal Pump Fault Identification Prototype

This prototype is aligned with the current Group 4 project workflow:
CSV/sensor data -> preprocessing -> FFT -> feature extraction -> ML -> SHAP/XAI ->
health score -> maintenance recommendation -> database/history -> dashboard.

## Important
The included demo dataset is synthetic. It is for testing the application interface and
software pipeline only. Do not present its accuracy as experimental pump results.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install packages:

   pip install -r requirements.txt

4. Run:

   streamlit run app.py

5. Open the local URL shown by Streamlit.

## CSV options

### A. Feature CSV
A labelled CSV can contain columns such as:

RMS, Peak, Peak-to-Peak, Std, Kurtosis, Skewness, Crest Factor,
Dominant Frequency, Temperature, Current, RPM, Fault

### B. Raw vibration CSV
A raw vibration file can contain a vibration/acceleration/signal column.
Optional columns:
Temperature, Current, RPM, Fault

The app then performs basic statistical feature extraction and FFT.

## Optional GenAI
Set:
- OPENAI_API_KEY
- OPENAI_MODEL

Example on Windows PowerShell:
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_MODEL="gpt-5.6-luna"

The GenAI feature is optional. SHAP/ML works without it.

## Suggested project evolution
Phase 1: CSV + Streamlit prototype
Phase 2: real experimental labelled data + validated ML
Phase 3: Raspberry Pi sensor ingestion
Phase 4: database/cloud synchronization
Phase 5: mobile/PWA interface
Phase 6: Digital Twin visualization and real-time notifications
