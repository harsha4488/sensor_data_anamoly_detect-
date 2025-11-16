"""
insights_nlp.py

Classical detectors + TF-IDF log correlation.
- robust preprocessing (force numeric conversion)
- zscore, iqr
- isolation_forest, local_outlier_factor (LOF)
- one_class_svm, elliptic_envelope
- dbscan
- pca_reconstruction (standard PCA)
- run_single_detector, run_all_detectors
- correlate_anomalies_with_logs
No LSTM here.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TF_AVAILABLE = False  # kept for compatibility

# -------------------------
# Preprocessing (robust)
# -------------------------
def preprocess_sensor_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_cols: Optional[List[str]] = None,
    resample_rule: Optional[str] = None,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    """
    - Ensures timestamp index (if present)
    - Converts all non-timestamp columns to numeric where possible
    - Drops fully-empty numeric columns
    """
    if df is None:
        raise ValueError("Input dataframe is None")

    df = df.copy()

    # convert timestamp column or index to datetime
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    else:
        # try index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # leave as-is; user may handle later
                pass

    # Force numeric conversion for all non-timestamp columns
    # If value_cols provided, convert only those
    cols_to_convert = value_cols if value_cols else [c for c in df.columns]
    for col in cols_to_convert:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            # remove common thousands separators and some units
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = df[col].str.replace("%", "", regex=False).str.replace("°", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select numeric columns
    if value_cols:
        # pick columns that exist and are numeric after conversion
        df_vals = df[[c for c in value_cols if c in df.columns]].select_dtypes(include=[np.number])
    else:
        df_vals = df.select_dtypes(include=[np.number])

    # Drop columns that are entirely NaN
    df_vals = df_vals.dropna(axis=1, how="all")

    if df_vals.shape[1] == 0:
        # Helpful error with hint to user
        raise ValueError(
            "No numeric columns found after conversion. "
            "Make sure your CSV contains at least one numeric column. "
            "Common issues: values with units (e.g. '72°C'), commas as thousands separators, extra quotes, or all values are non-numeric."
        )

    if resample_rule:
        df_vals = df_vals.resample(resample_rule).mean()

    if fill_method == "ffill":
        df_vals = df_vals.ffill().bfill()
    elif fill_method == "interpolate":
        df_vals = df_vals.interpolate().bfill()
    else:
        df_vals = df_vals.fillna(0)

    return df_vals

# -------------------------
# Numeric summary
# -------------------------
def numeric_summary(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if s.shape[0] else None,
        "std": float(s.std()) if s.shape[0] else None,
        "min": float(s.min()) if s.shape[0] else None,
        "50%": float(s.median()) if s.shape[0] else None,
        "max": float(s.max()) if s.shape[0] else None,
    }

# -------------------------
# Simple detectors
# -------------------------
def run_zscore(series: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    arr = series.ravel()
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr, dtype=bool)
    z = np.abs((arr - mean) / std)
    return z > z_thresh

def run_iqr(series: np.ndarray, k: float = 1.5) -> np.ndarray:
    arr = series.ravel()
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(arr, dtype=bool)
    return (arr < (q1 - k * iqr)) | (arr > (q3 + k * iqr))

# -------------------------
# ML detectors
# -------------------------
def run_isolation_forest(X: np.ndarray, contamination: float = 0.01, random_state: int = 42) -> np.ndarray:
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    clf.fit(X)
    preds = clf.predict(X)
    return preds == -1

def run_lof(X: np.ndarray, n_neighbors: int = 20, contamination: float = 0.01) -> np.ndarray:
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    preds = clf.fit_predict(X)
    return preds == -1

def run_oneclass_svm(X: np.ndarray, nu: float = 0.01, kernel: str = "rbf") -> np.ndarray:
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    clf.fit(X)
    preds = clf.predict(X)
    return preds == -1

def run_elliptic_envelope(X: np.ndarray, contamination: float = 0.01) -> np.ndarray:
    clf = EllipticEnvelope(contamination=contamination)
    clf.fit(X)
    preds = clf.predict(X)
    return preds == -1

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    clf = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clf.fit_predict(X)
    return labels == -1

def run_pca_reconstruction_error(X: np.ndarray, n_components: Optional[int] = None, multiplier: float = 3.0) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = min(n_features, max(1, n_features))
    n_components = min(n_components, n_features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(Xs)
    recon = pca.inverse_transform(proj)
    errors = np.mean((Xs - recon) ** 2, axis=1)
    thr = errors.mean() + multiplier * errors.std()
    return errors > thr

# -------------------------
# Dispatcher: single detector
# -------------------------
def run_single_detector(
    df_values: pd.DataFrame,
    detector_name: str,
    column: Optional[str] = None,
    n_components: Optional[int] = None,
) -> np.ndarray:
    """
    Run a single detector and return boolean mask length = n_rows.
    Only PCA uses n_components.
    """
    if df_values is None or df_values.shape[0] == 0:
        raise ValueError("df_values must be non-empty")

    if column:
        if column not in df_values.columns:
            raise ValueError(f"column {column} not found")
        X = df_values[[column]].values.astype(float)
    else:
        X = df_values.values.astype(float)

    vec = X[:, 0] if X.ndim == 2 else X.ravel()
    name = detector_name.lower()

    if name == "zscore":
        return run_zscore(vec)
    elif name == "iqr":
        return run_iqr(vec)
    elif name in ("isolation_forest", "isolationforest"):
        return run_isolation_forest(X)
    elif name in ("lof", "local_outlier_factor", "local outlier factor"):
        return run_lof(X)
    elif name in ("one_class_svm", "one-class-svm", "svm"):
        return run_oneclass_svm(X)
    elif name in ("elliptic_envelope", "elliptic"):
        return run_elliptic_envelope(X)
    elif name == "dbscan":
        return run_dbscan(X)
    elif name in ("pca_recon", "pca_reconstruction", "pca reconstruction"):
        return run_pca_reconstruction_error(X, n_components=n_components)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

# -------------------------
# Run all detectors
# -------------------------
def run_all_detectors(
    df_values: pd.DataFrame,
    column: Optional[str] = None,
    pca_n_components: Optional[int] = None
) -> Dict[str, Any]:
    if df_values is None or df_values.shape[0] == 0:
        raise ValueError("df_values must be non-empty")
    results: Dict[str, Any] = {}
    n_rows = df_values.shape[0]
    X = df_values.values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X) if X.size else X

    vec = Xs[:, 0] if Xs.ndim == 2 else Xs.ravel()

    results["zscore"] = run_zscore(vec)
    results["iqr"] = run_iqr(vec)

    try:
        results["isolation_forest"] = run_isolation_forest(Xs)
    except Exception:
        results["isolation_forest"] = np.zeros(n_rows, dtype=bool)

    try:
        results["local_outlier_factor"] = run_lof(Xs)
    except Exception:
        results["local_outlier_factor"] = np.zeros(n_rows, dtype=bool)

    try:
        results["one_class_svm"] = run_oneclass_svm(Xs)
    except Exception:
        results["one_class_svm"] = np.zeros(n_rows, dtype=bool)

    try:
        results["elliptic_envelope"] = run_elliptic_envelope(Xs)
    except Exception:
        results["elliptic_envelope"] = np.zeros(n_rows, dtype=bool)

    try:
        results["dbscan"] = run_dbscan(Xs)
    except Exception:
        results["dbscan"] = np.zeros(n_rows, dtype=bool)

    try:
        results["pca_recon"] = run_pca_reconstruction_error(Xs, n_components=pca_n_components)
    except Exception:
        results["pca_recon"] = np.zeros(n_rows, dtype=bool)

    results["meta"] = {"n_rows": n_rows, "n_cols": df_values.shape[1]}
    return results

# -------------------------
# TF-IDF log correlation
# -------------------------
def correlate_anomalies_with_logs(anomaly_indices: List[int], logs: List[str], top_k: int = 5) -> Dict[int, List[Tuple[int, float]]]:
    if not logs:
        return {}
    if not isinstance(logs, list):
        raise ValueError("logs must be a list of strings")
    vect = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf = vect.fit_transform(logs)
    out: Dict[int, List[Tuple[int, float]]] = {}
    n_logs = len(logs)
    for idx in anomaly_indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if i < 0 or i >= n_logs:
            continue
        sims = cosine_similarity(tfidf[i], tfidf).flatten()
        order = np.argsort(-sims)
        ranked = []
        for o in order:
            if o == i:
                continue
            ranked.append((int(o), float(sims[o])))
            if len(ranked) >= top_k:
                break
        out[i] = ranked
    return out

# -------------------------
# Utility (compat)
# -------------------------
def map_window_mask_to_index_mask(n_rows: int, window_size: int, window_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    if window_mask is None or window_mask.size == 0:
        return mask
    for i, v in enumerate(window_mask):
        if v:
            mask[i:i + window_size] = True
    return mask

# expose names
__all__ = [
    "preprocess_sensor_data",
    "numeric_summary",
    "run_single_detector",
    "run_all_detectors",
    "correlate_anomalies_with_logs",
    "map_window_mask_to_index_mask",
    "TF_AVAILABLE",
]
