
import numpy as np
import joblib
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# Bandpass filter
sos = butter(4, [0.5, 40], btype="bandpass",
             fs=300, output="sos")

# Feature extraction
def extract_features(signal):
    signal = sosfiltfilt(sos, signal)

    peaks, properties = find_peaks(
        signal, distance=90, prominence=0.45
    )

    rr_intervals = np.diff(peaks)

    if len(rr_intervals) > 0:
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
    else:
        mean_rr = 0
        std_rr = 0

    if len(rr_intervals) > 1:
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    else:
        rmssd = 0

    peak_count = len(peaks)
    signal_std = np.std(signal)

    return [mean_rr, std_rr, rmssd, peak_count, signal_std]


# Predict AF probabilities
def predict(signals, model_path=None):

    if model_path is None:
        model_path = Path(__file__).parent / "my_model.pkl"

    model = joblib.load(model_path)

    features = np.array([
        extract_features(signal) for signal in signals
    ])

    return model.predict_proba(features)[:, 1]
