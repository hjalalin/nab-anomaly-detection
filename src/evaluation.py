import numpy as np
import pandas as pd

def evaluate_predictions(df, preds, windows, early_tolerance="60min"):
    """
    Evaluate anomaly predictions against failure windows.
    
    Args:
        df (pd.DataFrame): Time-indexed dataframe.
        preds (array-like): Boolean mask or {0,1} predictions aligned with df.index.
        windows (list of (start, end)): Labeled failure windows as datetime tuples.
        early_tolerance (str or pd.Timedelta): Grace period before window start 
                                               where a detection counts as early warning.
    
    Returns:
        dict with counts and metrics (precision, recall, f1, coverage, early_warning_score).
    """
    preds = np.asarray(preds, dtype=bool)
    timestamps = df['timestamp'].to_numpy()

    if isinstance(early_tolerance, str):
        early_tolerance = pd.to_timedelta(early_tolerance)

    # Label mask (strict inside windows)
    label_mask = np.zeros(len(df), dtype=bool)
    for (s, e) in windows:
        label_mask |= (timestamps >= np.datetime64(s)) & (timestamps <= np.datetime64(e))

    # Compute TP/FP/FN/TN at point level
    tp = np.sum(preds & label_mask)
    fp = np.sum(preds & ~label_mask)
    fn = np.sum(~preds & label_mask)
    tn = np.sum(~preds & ~label_mask)

    # Standard metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Coverage (window-level hit ratio)
    window_hits = 0
    early_hits = 0
    n_windows = 0
    for (s, e) in windows:        # detections inside 
        
        if e < min(df['timestamp']):
            continue
        n_windows  += 1
        inside = (timestamps >= np.datetime64(s)) & (timestamps <= np.datetime64(e))
        # detections in early tolerance zone
        early_zone = (timestamps >= np.datetime64(s - early_tolerance)) & (timestamps < np.datetime64(s))

        if np.any(preds[early_zone]):
            early_hits += 1
            window_hits += 1  # count early detection as covering the window

        elif np.any(preds[inside]):
            window_hits += 1

    coverage = window_hits / n_windows if windows else 0.0
    early_detection_rate = early_hits / n_windows if windows else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "anomaly_window_detection_rate": coverage,
        "early_detection_rate": early_detection_rate
    }
