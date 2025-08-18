
import pandas as pd
import requests
from functools import lru_cache


def _labels_json():
    r = requests.get(LABELS_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def load_nab_series(file_name: str) -> pd.DataFrame:
    """
    Load a NAB time series CSV by file name.
    Returns a DataFrame indexed by 'timestamp' with a single 'value' column (or original columns).
    """
    url = BASE_SERIES_URL + file_name
    df = pd.read_csv(url)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    return df

def load_nab_labels(file_name: str):
    """
    Return labeled anomaly windows for the given file as a list of (start_ts, end_ts) Timestamps.
    Empty list if none.
    """
    labels = _labels_json()
    if file_name not in labels:
        return []
    return [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in labels[file_name]]
