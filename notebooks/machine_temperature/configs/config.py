import torch
# Central place for constants & defaults
# ---------------------------------------------------------------------

# Data sources
# ---------------------------------------------------------------------
DATASETS = {
    # Univariate examples - machine temperature
    "machine_temperature": {
        "files": [
            #"ambient_temperature_system_failure.csv",
            "machine_temperature_system_failure.csv"
            ],
        "labels_prefix": "realKnownCause/machine_temperature_system_failure.csv",
        "base_url": "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/realKnownCause/",
        "label_url": "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/labels/combined_windows.json"
    }
}

# ---------------------------------------------------------------------

# Defaults 
# ---------------------------------------------------------------------
DEFAULTS = {
    # --- Statistical: Global detectors ---
    "stats_global": {
        "z": {"enabled": True, "z_thresh": 3.0},
        "iqr": {"enabled": True, "k": 1.5},
        "mad": {"enabled": True,  "mad_z_thresh": 3.5}
    },

    # --- Statistical: Local detectors ---
    "stats_local": {
        "rolling_z": {"enabled": False, "window": 1000, "z_thresh": 3.0},
        "rolling_iqr": {"enabled": False, "window": 100, "k": 1.5},
        "rolling_mad": {"enabled": False, "window": 100, "mad_z_thresh": 3.5}, 
    },

    # --- Statistical: Sequential detectors ---
    "stats_sequential": {
        "cusum": {"enabled": True, "k": 10.0, "drift": 1.5},
        "ewma": {"enabled": True, "alpha": 1, "z_thresh": 3.0},
        "pct_change":  {"enabled": True,  "pct_thresh": 0.05, "pct_use_abs": True},
    },

    # --- Statistical: Trend detectors ---
    "stats_trend": {
        "mann_kendall": {"enabled": False, "alpha": 0.05, "window": 300},
    },

    # --- Forecasting detectors (placeholders if np/pd only) ---
    "forecast": {
        "arima": {"enabled": False, "window": 20, "z_thresh": 3.0},
        "prophet": {"enabled": False, "window": 20, "z_thresh": 3.0},
    },

    # --- Machine Learning detectors (not active) ---
    "ml": {
        "isolation_forest": {"enabled": False, "contamination": 0.01},
    },

    # --- Deep Learning (Autoencoders etc.) ---
    "dl": {
        "autoencoder": {
            "enabled": False,
            "seq_len": 50,
            "hidden": 64,
            "layers": 2,
            "dropout": 0.1,
            "batch_size": 32,
            "weight_decay": 0.0, 
            "lr": 1e-3,
            "epochs": 50,
            "patience": 5,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
    },

    # --- Evaluation / Aggregation ---
    "aggregate": {"mode": "any"},   # {"any","all","majority"}
    "threshold": {"quantile": 0.995},
}

