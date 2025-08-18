import pandas as pd
import requests
from config import BASE_SERIES_URL, LABELS_URL

def load_nab_series(file_name: str = 'machine_temperature_systemB_failure.csv') -> pd.DataFrame:
    """
    Load a NAB time series CSV by file name from ASE_SERIES_URL.
    file_name='machine_temperature_systemB_failure.csv'
    """
    url = BASE_SERIES_URL.rstrip("/") + "/" + file_name
    df = pd.read_csv(url)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def load_labels_json(label_url = LABELS_URL):
    """Fetch the NAB combined windows JSON from LABELS_URL."""
    response = requests.get(label_url)
    return response.json()

def load_nab_anomaly_windows(file_name="machine_temperature_system_failure.csv"):
    """
    Load labeled anomaly windows for a given file from LABELS_URL.
    Uses label key format: 'realKnownCause/<file_name>'
    """
    labels = _loadlabels_json()
    label_key = f"realKnownCause/{file_name}"
    if label_key not in labels:
        return []
    return [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in labels[label_key]]


def load_data_and_labels():
    """
    Convenience: returns (df, anomaly_windows) in one call.
    """
    df = load_nab_series()
    windows = load_nab_anomaly_windows()
    return df, windows
