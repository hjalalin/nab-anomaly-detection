# Central place for constants & defaults
# ---------------------------------------------------------------------

# Data sources
# ---------------------------------------------------------------------
BASE_SERIES_URL = (
    "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/realKnownCause/"
)
LABELS_URL = (
    "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/labels/combined_windows.json"
)
# ---------------------------------------------------------------------

# Defaults 
# ---------------------------------------------------------------------
DEFAULTS = {
    # --- Statistical: Global detectors ---
    "Z_K": 3.0,                 # Z-score k
    "IQR_K": 1.5,               # IQR multiplier
    "MAD_K": 3.5,               # MAD multiplier

    # --- Statistical: Local detectors ---
    "ROLLING_WINDOW": 100,      # window length
    "ROLLING_K": 3.0,           # k×σ for rolling mean/std

    # --- Statistical: Sequential detectors ---
    "PCT_CHANGE": 0.05,         # 5% relative change
    "CUSUM_THRESHOLD": 5.0,     # CUSUM detection threshold
    "CUSUM_DRIFT": 0.0,         # CUSUM drift

    # --- Statistical: Trend detectors ---
    "MK_ALPHA": 0.05,           # significance

    # --- Machine Learning detectors ---
    "ISOF_CONTAM": 0.01,        # Isolation Forest contamination

    # --- Deep Learning (Autoencoders) ---
    "SEQ_LEN": 100,             # window length for training
    "AE_HIDDEN": 64,            # hidden dimension
    "AE_LAYERS": 2,             # encoder/decoder layers
    "AE_DROPOUT": 0.1,          # dropout
    "TRAIN_EPOCHS": 50,         # default training epochs
    "BATCH_SIZE": 32,

    # --- Evaluation / Thresholding ---
    "THRESHOLD_Q": 0.995       # quantile for anomaly threshold
}
