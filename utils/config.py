# Central place for constants & defaults
# ---------------------------------------------------------------------

# Data sources
# ---------------------------------------------------------------------
DATASETS = {
    # Univariate examples - machine temperature
    "machine_temperature": {
        "files": [
            "ambient_temperature_system_failure.csv",
            "cpu_utilization_asg_misconfiguration.csv",
            "ec2_request_latency_system_failure.csv",
            "machine_temperature_system_failure.csv",
            "nyc_taxi.csv",
            "rogue_agent_key_hold.csv",
            "rogue_agent_key_updown.csv"
            ],
        "labels_prefix": "realKnownCause/",
        "base_url": "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/realKnownCause/",
        "label_url": "https://raw.githubusercontent.com/hjalalin/nab-anomaly-detection/main/data/labels/combined_windows.json"
    },
    # Multivariate (CloudWatch) 
    "aws_cloudwatch_cpu_net": {
        "files": [],
        "labels_prefix": "realAWSCloudwatch/",
        "base_url": "",
        "label_url": "",
    },
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
        "rolling_z": {"enabled": True, "window": 100, "z_thresh": 3.0},
        "rolling_iqr": {"enabled": False, "window": 100, "k": 1.5},
        "rolling_mad": {"enabled": False, "window": 100, "mad_z_thresh": 3.5}, 
    },

    # --- Statistical: Sequential detectors ---
    "stats_sequential": {
        "cusum": {"enabled": True, "k": 5.0, "drift": 0.0},
        "ewma": {"enabled": False, "alpha": 0.2, "z_thresh": 3.0},
        "pct_change":  {"enabled": True,  "pct_thresh": 0.05, "pct_use_abs": True},
    },

    # --- Statistical: Trend detectors ---
    "stats_trend": {
        "mann_kendall": {"enabled": True, "alpha": 0.05},
    },

    # --- Forecasting detectors (placeholders if np/pd only) ---
    "forecast": {
        "arima": {"enabled": False, "window": 20, "z_thresh": 3.0},
        "prophet": {"enabled": False, "window": 20, "z_thresh": 3.0},
    },

    # --- Machine Learning detectors (not active in np/pd-only version) ---
    "ml": {
        "isolation_forest": {"enabled": False, "contamination": 0.01},
    },

    # --- Deep Learning (Autoencoders etc.) ---
    "dl": {
        "autoencoder": {
            "enabled": False,
            "seq_len": 100,
            "hidden": 64,
            "layers": 2,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 32,
        },
    },

    # --- Evaluation / Aggregation ---
    "aggregate": {"mode": "any"},   # {"any","all","majority"}
    "threshold": {"quantile": 0.995},
}

