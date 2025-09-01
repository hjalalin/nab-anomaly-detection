import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 0) Helpers
# -----------------------------
def shade_windows(ax, windows, color="red", alpha=0.3, first_label="Failure Window"):
    if not windows:
        return
    for i, (s, e) in enumerate(windows):
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)
        ax.axvspan(s, e, color=color, alpha=alpha,
                   label=first_label if i == 0 else "_nolegend_")


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
def plot_full_series_with_windows(df,windows=None,title="", ylabel="",xlabel="Date",figsize=(14, 4),grid=True,subplots=False):
    """   Plots all columns in `df` over time and Shades given anomaly/failure windows.    """
    cols = [c for c in df.columns if c.lower() != "timestamp"]
    if subplots and len(cols) > 1:
        fig, axes = plt.subplots(len(df.columns), 1, figsize=(figsize[0], figsize[1]*len(df.columns)), sharex=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.plot(df["timestamp"], df[col], label=col)
            if windows is not None:
                shade_windows(ax, windows, color="red", alpha=0.3, first_label="Failure Time")
            ax.set_title(f"{title} - {col}")
            ax.set_ylabel(ylabel)
            if grid:
                ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel(xlabel)
        fig.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for col in cols:
            ax.plot(df["timestamp"], df[col], label=col)
        if windows is not None:
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
# 2) Multi-subplot comparison for multiple methods
# -----------------------------
def plot_methods_subplots(
    df,
    methods_preds,
    value_col,
    offset = 0,
    windows=None,
    sharex=True,
    figsize=None,
    max_cols=2,
    marker="o",
    grid=True,
):
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

    x = df['timestamp']
    y = df[value_col]
    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(x, y, lw=1, label="Series")
        shade_windows(ax, windows, color="red", alpha=0.25, first_label="Failure Time")
        idx = _indices_from_preds(methods_preds[name], len(df))
        if len(idx) > 0:
            ax.scatter(x[idx + offset], y[idx+offset], marker=marker, color='red', label=f'Detected Anomalies')
        if len(names)>0:
            ax.set_title(name)
        if grid:
            ax.grid(True)
        ax.legend(loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.xlim(min(df['timestamp']), max(df['timestamp']))

    fig.tight_layout()
    plt.show()


