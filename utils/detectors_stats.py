# utils/apply_detectors.py
import numpy as np
import pandas as pd


# ---------- Helpers ----------

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd

def _iqr_bounds(x: np.ndarray, k: float = 1.5):
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def _mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation (MAD), scaled so that
    sigma ~= 1.4826 * MAD for normal data.
    Returns the *unscaled* MAD; caller applies scaling.
    """
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def _modified_zscores(x: np.ndarray) -> np.ndarray:
    """
    Modified z-score using MAD:
    mz = 0.6745 * (x - median) / MAD
    If MAD==0 -> zeros.
    """
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = _mad(x)
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad

def _rolling_apply(x: pd.Series, window: int, fn):
    return x.rolling(window=window, min_periods=max(3, window // 3)).apply(fn, raw=False)


# ---------- Global ----------

def global_z(x, z_thresh=3.0):
    z = _zscore(x)
    mask = np.abs(z) >= z_thresh
    return {"mask": mask, "score": z}

def global_iqr(x, k=1.5):
    lo, hi = _iqr_bounds(x, k=k)
    mask = (x < lo) | (x > hi)
    return {"mask": mask, "score": mask.astype(float)}

def global_mad(x, z_thresh=3.5):
    """
    Robust anomaly detection using modified z-scores.
    Typical threshold ~3.5.
    """
    mz = _modified_zscores(np.asarray(x, float))
    mask = np.abs(mz) >= z_thresh
    return {"mask": mask, "score": mz}


# ---------- Local ----------

def rolling_z(x: pd.Series, window=60, z_thresh=3.0):
    def _last_z(s: pd.Series):
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return 0.0
        return (s.iloc[-1] - s.mean()) / sd
    z = _rolling_apply(x, window, _last_z).fillna(0).to_numpy()
    mask = np.abs(z) >= z_thresh
    return {"mask": mask, "score": z}

def rolling_iqr(x: pd.Series, window=60, k=1.5):
    def _last_outlier(s: pd.Series):
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        v = s.iloc[-1]
        return 1.0 if (v < lo or v > hi) else 0.0
    out = _rolling_apply(x, window, _last_outlier).fillna(0).to_numpy()
    return {"mask": out.astype(bool), "score": out}

def rolling_mad(x: pd.Series, window=60, z_thresh=3.5):
    """
    Windowed modified z-score on the last sample of each window.
    """
    def _last_mz(s: pd.Series):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return 0.0
        return 0.6745 * (s.iloc[-1] - med) / mad
    mz = _rolling_apply(x, window, _last_mz).fillna(0).to_numpy()
    mask = np.abs(mz) >= z_thresh
    return {"mask": mask, "score": mz}

def pct_change(x: pd.Series, pct_thresh=0.2, use_abs=True):
    """
    Point-to-point percentage change (x_t vs x_{t-1}).
    pct_thresh = 0.2 -> 20%.
    """
    pc = x.pct_change().fillna(0.0).to_numpy()
    if use_abs:
        mask = np.abs(pc) >= pct_thresh
    else:
        # Only flag positive jumps >= pct_thresh
        mask = pc >= pct_thresh
    return {"mask": mask, "score": pc}

def rolling_pct_change(x: pd.Series, window=60, pct_thresh=0.2, ref="median"):
    """
    Percent change of the last value in the window vs a reference statistic
    (median or mean) of that window.
    """
    ref = str(ref).lower()
    def _last_pc(s: pd.Series):
        v = s.iloc[-1]
        r = s.median() if ref == "median" else s.mean()
        denom = np.abs(r)
        if denom == 0 or np.isnan(denom):
            return 0.0
        return (v - r) / denom
    pc = _rolling_apply(x, window, _last_pc).fillna(0.0).to_numpy()
    mask = np.abs(pc) >= pct_thresh
    return {"mask": mask, "score": pc}


# ---------- Sequential ----------

def cusum(x, k=5.0, drift=0.0):
    z = _zscore(x)
    cpos = np.zeros_like(z)
    cneg = np.zeros_like(z)
    for i in range(1, len(z)):
        cpos[i] = max(0.0, cpos[i-1] + z[i] - drift)
        cneg[i] = min(0.0, cneg[i-1] + z[i] + drift)
    mask = (cpos >= k) | (-cneg >= k)
    return {"mask": mask, "score": cpos - cneg}

def ewma_ph(x, alpha=0.2, z_thresh=3.0):
    x = np.asarray(x, float)
    ewma = pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    resid = x - ewma
    z = _zscore(resid)
    mask = np.abs(z) >= z_thresh
    return {"mask": mask, "score": z}


# ---------- Trend ----------

def mann_kendall(x, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return {"tau": 0.0, "p_value": 1.0, "trend": "no trend"}
    s = 0
    for k in range(n-1):
        s += np.sum(np.sign(x[k+1:] - x[k]))
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts-1) * (counts+1))
    var_s = (n*(n-1)*(2*n+5) - tie_term)/18.0
    if var_s == 0:
        return {"tau": 0.0, "p_value": 1.0, "trend": "no trend"}
    z = (s-1)/np.sqrt(var_s) if s>0 else (s+1)/np.sqrt(var_s) if s<0 else 0.0
    from math import erf, sqrt
    phi = 0.5*(1.0+erf(abs(z)/sqrt(2.0)))
    p = 2.0*(1.0-phi)
    tau = (2*s)/(n*(n-1))
    trend = "no trend"
    if p < alpha:
        trend = "increasing" if tau > 0 else "decreasing"
    return {"tau": tau, "p_value": p, "trend": trend}



# ---------- Orchestrator ----------

def run_detectors(df: pd.DataFrame, value_col: str, config: dict):
    """
    Apply detectors based on config from utils/config.
    Returns dict with per-method results and combined mask.
    """
    x = df[value_col].to_numpy()
    s = df[value_col]

    per_method = {}
    masks = []

    # Global
    if config["stats_global"].get("z", False):
        res = global_z(x, config["stats_global"].get("z_thresh", 3.0))
        per_method["global_z"] = res; masks.append(res["mask"])
    if config["stats_global"].get("iqr", False):
        res = global_iqr(x, config["stats_global"].get("iqr_k", 1.5))
        per_method["global_iqr"] = res; masks.append(res["mask"])
    if config["stats_global"].get("mad", False):
        res = global_mad(x, config["stats_global"].get("mad_z_thresh", 3.5))
        per_method["global_mad"] = res; masks.append(res["mask"])

    # Local
    if config["stats_local"].get("rolling_z", False):
        res = rolling_z(s, config["stats_local"].get("window", 60), config["stats_local"].get("z_thresh", 3.0))
        per_method["rolling_z"] = res; masks.append(res["mask"])
    if config["stats_local"].get("rolling_iqr", False):
        res = rolling_iqr(s, config["stats_local"].get("window", 60), config["stats_local"].get("k", 1.5))
        per_method["rolling_iqr"] = res; masks.append(res["mask"])
    if config["stats_local"].get("rolling_mad", False):
        res = rolling_mad(s, config["stats_local"].get("window", 60), config["stats_local"].get("mad_z_thresh", 3.5))
        per_method["rolling_mad"] = res; masks.append(res["mask"])
    if config["stats_local"].get("rolling_pct_change", False):
        res = rolling_pct_change(
            s,
            config["stats_local"].get("window", 60),
            config["stats_local"].get("pct_thresh", 0.2),
            config["stats_local"].get("pct_ref", "median"),
        )
        per_method["rolling_pct_change"] = res; masks.append(res["mask"])

    # Sequential
    if config["stats_sequential"].get("cusum", False):
        res = cusum(x, config["stats_sequential"].get("k", 5.0), config["stats_sequential"].get("drift", 0.0))
        per_method["cusum"] = res; masks.append(res["mask"])
    if config["stats_sequential"].get("ewma", False):
        res = ewma_ph(x, config["stats_sequential"].get("alpha", 0.2), config["stats_sequential"].get("z_thresh", 3.0))
        per_method["ewma"] = res; masks.append(res["mask"])
    if config["stats_sequential"].get("pct_change", False):
        res = pct_change(s, config["stats_sequential"].get("pct_thresh", 0.2), config["stats_sequential"].get("pct_use_abs", True))
        per_method["pct_change"] = res; masks.append(res["mask"])

    # Trend
    if config["stats_trend"].get("mann_kendall", False):
        per_method["mann_kendall"] = mann_kendall(x, config["stats_trend"].get("alpha", 0.05))

    # Combine
    combined = np.column_stack(masks).any(axis=1) if masks else np.zeros(len(x), bool)

    return {"per_method": per_method, "combined_mask": combined}

