import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 0) Helpers
# -----------------------------
def shade_windows(ax, windows, color="red", alpha=0.3, first_label="Anomaly window"):
    if not windows:
        return
    for i, (s, e) in enumerate(windows):
        ax.axvspan(s, e, color=color, alpha=alpha, label=first_label if i == 0 else "_nolegend_")


def _indices_from_preds(preds, n):
    """
    Convert predictions to anomaly indices.
    - If preds is None: return empty list
    - If preds is a list/array of 0/1: return indices where ==1
    - If preds is a list/array of ints: treat them as anomaly indices
    """
    if preds is None:
        return []
    preds = np.asarray(preds)
    if preds.dtype == bool or set(np.unique(preds)).issubset({0, 1}):
        return np.where(preds == 1)[0]
    return preds.astype(int)


# -----------------------------
# 1) Full series with shaded windows
# -----------------------------
def plot_full_series_with_windows(
    df,
    value_col="value",
    windows=None,
    title="Series with Labeled Anomalies",
    series_label="Series",
    ylabel="",
    xlabel="Timestamp",
    figsize=(14, 4),
    grid=True,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(df.index, df[value_col], label=series_label)
    shade_windows(ax, windows, color="red", alpha=0.3, first_label="Failure Time")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if grid:
        ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


# -----------------------------
# 2) Overlay detections on top of shaded windows
# -----------------------------
def plot_series_with_windows_and_points(
    df,
    value_col="value",
    preds=None,
    windows=None,
    title="Detected Anomalies",
    series_label="Series",
    point_label="Detected anomaly",
    figsize=(14, 4),
    marker="x",
    grid=True,
):
    idx = _indices_from_preds(preds, len(df))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(df.index, df[value_col], label=series_label)
    shade_windows(ax, windows, color="red", alpha=0.3, first_label="Failure Time")
    if len(idx) > 0:
        ax.scatter(df.index[idx], df[value_col].iloc[idx], marker=marker, label=point_label)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    if grid:
        ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


# -----------------------------
# 3) Multi-subplot comparison for multiple methods
# -----------------------------
def plot_methods_subplots(
    df,
    methods_preds,
    value_col="value",
    windows=None,
    sharex=True,
    figsize=None,
    max_cols=2,
    marker="x",
    grid=True,
):
    """
    methods_preds: {'Z-Score': preds_z, 'ARIMA': preds_arima, ...}
    """
    names = list(methods_preds.keys())
    n = len(names)
    if n == 0:
        raise ValueError("methods_preds is empty.")
    if figsize is None:
        rows = int(np.ceil(n / max_cols))
        figsize = (14, 3 * rows)

    rows = int(np.ceil(n / max_cols))
    cols = min(max_cols, n)

    fig, axes = plt.subplots(rows, cols, sharex=sharex, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    x = df.index
    y = df[value_col]
    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(x, y, lw=1, label="Series")
        shade_windows(ax, windows, color="red", alpha=0.25, first_label="Failure Time")
        idx = _indices_from_preds(methods_preds[name], len(df))
        if len(idx) > 0:
            ax.scatter(x[idx], y.iloc[idx], marker=marker, label=name)
        ax.set_title(name)
        if grid:
            ax.grid(True)
        ax.legend(loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    plt.show()


# -----------------------------
# 4) Error + threshold plot
# -----------------------------
def plot_error_and_threshold(errors, thr, title="Error & Threshold", figsize=(14, 3), grid=True):
    errors = np.asarray(errors)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(errors, lw=1)
    ax.axhline(thr, color="red", linestyle="--", label="Threshold")
    ax.set_title(title)
    if grid:
        ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()

