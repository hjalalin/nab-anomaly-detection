import pandas as pd
import requests
from utils.config import DATASETS 


def load_labels_json(label_url):
    response = requests.get(label_url)
    return response.json()


def load_failure_windows(label_key, label_url):
    labels = load_labels_json(label_url)
    if label_key not in labels: return []
    return [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in labels[label_key]]


def load_dataset(name: str = 'machine_temperature') -> pd.DataFrame:
    cfg = DATASETS[name]
    base_url = cfg["base_url"].rstrip("/") + "/"

    if cfg["type"] == "univariate":
        df = pd.read_csv(base_url + cfg["file"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        return df, load_failure_windows(cfg["labels_prefix"] + cfg["file"], cfg["label_url"])

     # multivariate: list of files → merge on timestamp
    dfs = []
    for f in cfg["files"]:
        df = pd.read_csv(base_url + f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        val_col = [c for c in df.columns if c != "timestamp"][0]
        df = df.rename(columns={val_col: f.split(".")[0]})
        dfs.append(df)
    df = dfs[0]
    for d in dfs[1:]:
        df = pd.merge(df, d, on="timestamp", how="inner")
    df = df.set_index("timestamp")

    windows = load_failure_windows(cfg["labels_prefix"] + cfg["files"][0], cfg["label_url"])
    return df, windows


