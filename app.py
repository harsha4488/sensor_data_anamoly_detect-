# app.py
"""
Streamlit app:
- Multi-model anomaly detection (IsolationForest, Z-Score, OCSVM, LOF, Elliptic, PCA, IQR, LSTM Autoencoder)
- Robust TF-IDF log correlation
- Lightweight log analysis toolkit (timeline, top ngrams, topics via NMF, clustering)
- Direct Groq LLM chat (optional — paste key in sidebar to enable)
- Context-aware chat: LLM receives sensor/log/correlation context when answering
- Generate & download synthetic 1-month minute-level sensor + log CSVs
- Highlight anomaly rows in the sensor table
"""
import json
import io
import os
import re
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Optional LLM (Groq via langchain_groq)
_HAS_GROQ = False
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    _HAS_GROQ = True
except Exception:
    _HAS_GROQ = False

# Keras import strategy (LSTM autoencoder optional)
layers = None
models = None
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras import layers, models
except Exception:
    try:
        import keras  # noqa: F401
        from keras import layers, models  # type: ignore
    except Exception:
        layers = None
        models = None

# -------------------------
# Utilities
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    toks = re.findall(r"[a-zA-Z0-9]+", text)
    return [t for t in toks if len(t) > 1]

def preprocess_text_for_tfidf(texts: List[str]) -> List[str]:
    return [" ".join(simple_tokenize(t)) for t in texts]

# -------------------------
# Anomaly detectors
# -------------------------
def detect_anomalies_isolation(df: pd.DataFrame, contamination=0.01) -> pd.Series:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    X = StandardScaler().fit_transform(num.fillna(num.mean()))
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    return pd.Series(preds == -1, index=df.index)

def detect_anomalies_zscore(df: pd.DataFrame, z_thresh=3.5, window: Optional[int]=None) -> pd.Series:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    if window and window > 1:
        roll_mean = num.rolling(window, center=True, min_periods=1).mean()
        roll_std = num.rolling(window, center=True, min_periods=1).std().replace(0, 1)
        z = (num - roll_mean) / roll_std
    else:
        z = (num - num.mean()) / num.std().replace(0, 1)
    return (z.abs() > z_thresh).any(axis=1)

def detect_anomalies_ocsvm(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    model = OneClassSVM(nu=0.05, kernel="rbf")
    preds = model.fit_predict(num)
    return pd.Series(preds == -1, index=df.index)

def detect_anomalies_lof(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    preds = model.fit_predict(num)
    return pd.Series(preds == -1, index=df.index)

def detect_anomalies_elliptic(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    model = EllipticEnvelope(contamination=0.05)
    preds = model.fit_predict(num)
    return pd.Series(preds == -1, index=df.index)

def detect_anomalies_pca(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    pca = PCA(n_components=0.95)
    Xp = pca.fit_transform(num)
    recon = pca.inverse_transform(Xp)
    mse = ((num - recon) ** 2).mean(axis=1)
    thr = mse.mean() + 3 * mse.std()
    return pd.Series(mse > thr, index=df.index)

def detect_anomalies_iqr(df: pd.DataFrame, window=48, factor=1.5) -> pd.Series:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.Series([False]*len(df), index=df.index)
    flags = pd.Series(False, index=df.index)
    for col in num.columns:
        q1 = num[col].rolling(window, min_periods=1).quantile(0.25)
        q3 = num[col].rolling(window, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        flags |= (num[col] < lower) | (num[col] > upper)
    return flags

# LSTM autoencoder (optional)
def build_lstm_autoencoder(seq_len:int, nfeat:int):
    if models is None or layers is None:
        raise RuntimeError("Keras/TensorFlow not available.")
    m = models.Sequential([
        layers.Input(shape=(seq_len, nfeat)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16, return_sequences=False),
        layers.RepeatVector(seq_len),
        layers.LSTM(16, return_sequences=True),
        layers.LSTM(32, return_sequences=True),
        layers.TimeDistributed(layers.Dense(nfeat))
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def detect_anomalies_lstm_autoencoder(df: pd.DataFrame, seq_len=24, epochs=5) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).fillna(0)
    if num.shape[1] == 0 or len(num) < seq_len + 1:
        return pd.Series([False]*len(df), index=df.index)
    Xvals = num.values
    seqs = [Xvals[i:i+seq_len] for i in range(len(Xvals)-seq_len)]
    X = np.array(seqs)
    train_size = int(0.8 * len(X))
    if train_size < 1:
        return pd.Series([False]*len(df), index=df.index)
    X_train, X_test = X[:train_size], X[train_size:]
    model = build_lstm_autoencoder(seq_len, num.shape[1])
    model.fit(X_train, X_train, epochs=epochs, batch_size=32, verbose=0)
    pred = model.predict(X_test)
    mse = np.mean((X_test - pred)**2, axis=(1,2))
    thr = mse.mean() + 3 * mse.std()
    flags = mse > thr
    full = [False]*len(df)
    for i, f in enumerate(flags):
        pos = train_size + i + seq_len
        if pos < len(full):
            full[pos] = bool(f)
    return pd.Series(full, index=df.index)

# -------------------------
# Signature + robust correlation
# -------------------------
def anomaly_signature(df: pd.DataFrame, row_idx:int, top_k:int=3) -> str:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return "no-numeric-features"
    row = num.iloc[row_idx]
    z = ((row - num.mean()) / num.std().replace(0,1)).abs().sort_values(ascending=False)
    feats = z.head(top_k).index.tolist()
    return " ; ".join([f"{f}" for f in feats])

def correlate_anomalies_with_logs(sensor_df: pd.DataFrame, logs_df: pd.DataFrame,
                                  anomaly_series: pd.Series, window_minutes:int=30, top_matches:int=3) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = []
    logs = logs_df.copy()
    if "timestamp" in logs.columns:
        try:
            logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
            logs = logs.set_index("timestamp").sort_index()
        except Exception:
            pass
    else:
        try:
            logs.index = pd.to_datetime(logs.index)
            logs = logs.sort_index()
        except Exception:
            pass

    if "log" not in logs.columns:
        raise ValueError("Logs dataframe must contain a 'log' column.")

    logs["_preproc_text"] = preprocess_text_for_tfidf(logs["log"].astype(str).tolist())
    corpus_texts = logs["_preproc_text"].fillna("").tolist()
    tfidf = None
    logs_vec = None
    if any(corpus_texts):
        tfidf = TfidfVectorizer(max_features=2000)
        logs_vec = tfidf.fit_transform(corpus_texts)

    index_to_pos = {}
    for p, idx in enumerate(logs.index):
        if idx not in index_to_pos:
            index_to_pos[idx] = p

    anomaly_positions = np.where(anomaly_series)[0]
    sensor_idx = sensor_df.index if isinstance(sensor_df.index, pd.DatetimeIndex) else None

    for pos in anomaly_positions:
        ts = None
        try:
            ts = sensor_df.index[pos] if sensor_idx is not None else None
        except Exception:
            ts = None

        signature = anomaly_signature(sensor_df, pos)
        sig_proc = " ".join(simple_tokenize(signature))

        if ts is not None and isinstance(logs.index, pd.DatetimeIndex):
            start = ts - pd.Timedelta(minutes=window_minutes)
            end = ts + pd.Timedelta(minutes=window_minutes)
            try:
                window_logs = logs.loc[start:end]
            except Exception:
                window_logs = logs
        else:
            window_logs = logs

        if window_logs.empty:
            results.append({
                "anomaly_pos": int(pos),
                "timestamp": str(ts),
                "signature": signature,
                "matches": []
            })
            continue

        window_positions = []
        for idx in window_logs.index:
            p = index_to_pos.get(idx, None)
            if p is not None:
                window_positions.append(p)

        matches = []
        if logs_vec is not None and tfidf is not None and len(window_positions) > 0:
            sig_vec = tfidf.transform([sig_proc])
            candidate_vecs = logs_vec[window_positions]
            sims = cosine_similarity(sig_vec, candidate_vecs).flatten()
            ranked_idx = np.argsort(-sims)[:top_matches]
            for r in ranked_idx:
                score = float(sims[r])
                if score <= 0:
                    continue
                global_pos = window_positions[r]
                original_text = logs["log"].iloc[global_pos]
                matches.append({
                    "log_pos": int(global_pos),
                    "log_index": str(logs.index[global_pos]),
                    "timestamp": str(logs.index[global_pos]),
                    "text": original_text,
                    "score": round(score, 4)
                })
        else:
            sig_set = set(sig_proc.split())
            for idx, row in window_logs.iterrows():
                txt = row["_preproc_text"]
                txt_set = set(txt.split())
                if not txt_set:
                    continue
                overlap = len(sig_set & txt_set) / max(1, len(sig_set | txt_set))
                if overlap > 0:
                    matches.append({
                        "log_index": str(idx),
                        "timestamp": str(idx),
                        "text": row.get("log", "")[:400],
                        "score": round(float(overlap), 4)
                    })
            matches = sorted(matches, key=lambda x: -x["score"])[:top_matches]

        results.append({
            "anomaly_pos": int(pos),
            "timestamp": str(ts),
            "signature": signature,
            "matches": matches
        })

    return results

# -------------------------
# Log analysis toolkit
# -------------------------
_STOPWORDS = set("""
the and is to of in for on with at as by this that from or it be are was were have has had not but
""".split())

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', ' ', t)
    t = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', ' ', t)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def parse_timestamps(df, ts_col_candidates=("timestamp","time","ts")):
    df2 = df.copy()
    for c in ts_col_candidates:
        if c in df2.columns:
            try:
                df2[c] = pd.to_datetime(df2[c], errors='coerce')
                if df2[c].notna().sum() > 0:
                    df2 = df2.set_index(c).sort_index()
                    return df2, c
            except Exception:
                pass
    try:
        idx = pd.to_datetime(df2.index, errors='coerce')
        if idx.notna().sum() > 0:
            df2.index = idx
            return df2, None
    except Exception:
        pass
    return df2, None

def top_ngrams(texts, n=20, ngram_range=(1,1), stopwords=_STOPWORDS):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=list(stopwords))
    X = vect.fit_transform(texts)
    counts = X.sum(axis=0).A1
    terms = vect.get_feature_names_out()
    top_idx = counts.argsort()[::-1][:n]
    return [(terms[i], int(counts[i])) for i in top_idx]

def extract_severity_counts(texts):
    counters = {"error":0,"warning":0,"critical":0,"info":0}
    for t in texts:
        tl = t.lower()
        if "error" in tl or "failed" in tl or "exception" in tl:
            counters["error"] += 1
        if "warn" in tl or "warning" in tl:
            counters["warning"] += 1
        if "critical" in tl:
            counters["critical"] += 1
        if "info" in tl or "started" in tl:
            counters["info"] += 1
    return counters

def build_timeline(df_index, freq="1H"):
    try:
        s = pd.Series(1, index=pd.to_datetime(df_index))
        ts = s.resample(freq).sum().fillna(0).astype(int)
        return ts
    except Exception:
        return pd.Series(dtype=int)

def nmf_topics(texts, n_topics=4, n_top_words=6):
    tfidf = TfidfVectorizer(max_features=2000, stop_words=_STOPWORDS)
    X = tfidf.fit_transform(texts)
    if X.shape[0] < n_topics:
        n_topics = max(1, X.shape[0]//1)
    nmf = NMF(n_components=n_topics, random_state=0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = tfidf.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[::-1][:n_top_words]
        topics.append([terms[i] for i in top_indices])
    return topics, W

def cluster_logs(texts, n_clusters=4):
    tfidf = TfidfVectorizer(max_features=2000, stop_words=_STOPWORDS)
    X = tfidf.fit_transform(texts)
    if X.shape[0] < n_clusters:
        n_clusters = max(1, X.shape[0])
    k = KMeans(n_clusters=n_clusters, random_state=0)
    labels = k.fit_predict(X)
    centers = k.cluster_centers_
    Xn = normalize(X)
    cn = normalize(centers)
    reps = {}
    for c in range(n_clusters):
        sims = (Xn @ cn[c].T).A1 if hasattr(Xn, "A1") else (Xn @ cn[c].T)
        best = int(sims.argmax())
        reps[c] = best
    return labels, reps

def analyze_logs(logs_df, text_col="log", ts_candidates=("timestamp","time","ts"), top_n=20, timeline_freq="1H"):
    out = {}
    if text_col not in logs_df.columns:
        raise ValueError(f"logs_df must contain a '{text_col}' column")
    df_ts, ts_col = parse_timestamps(logs_df, ts_col_candidates=ts_candidates)
    texts_raw = df_ts[text_col].astype(str).fillna("").tolist()
    cleaned = [clean_text(t) for t in texts_raw]
    out["cleaned_texts"] = cleaned
    out["df_with_index"] = df_ts
    out["timestamp_col"] = ts_col
    out["severity_counts"] = extract_severity_counts(texts_raw)
    out["top_unigrams"] = top_ngrams(cleaned, n=top_n, ngram_range=(1,1))
    out["top_bigrams"] = top_ngrams(cleaned, n=top_n, ngram_range=(2,2))
    out["timeline"] = build_timeline(df_ts.index, freq=timeline_freq)
    try:
        topics, nmf_W = nmf_topics(cleaned, n_topics=min(6, max(1, len(cleaned)//5)))
        out["topics"] = topics
        out["nmf_weights"] = nmf_W
    except Exception:
        out["topics"] = []
        out["nmf_weights"] = None
    try:
        labels, reps = cluster_logs(cleaned, n_clusters=min(6, max(1, len(cleaned)//5)))
        out["cluster_labels"] = labels
        out["cluster_representatives"] = reps
    except Exception:
        out["cluster_labels"] = None
        out["cluster_representatives"] = None
    counter = {}
    for t in cleaned:
        counter[t] = counter.get(t,0) + 1
    out["most_common_cleaned"] = sorted(counter.items(), key=lambda x:-x[1])[:10]
    total = len(cleaned)
    matched = out["severity_counts"].get("error",0) + out["severity_counts"].get("critical",0)
    lines = []
    lines.append(f"Analyzed {total} log entries. Detected {matched} error/critical mentions.")
    if out["topics"]:
        lines.append("Top topics (words):")
        for i, t in enumerate(out["topics"][:5]):
            lines.append(f"  Topic {i+1}: " + ", ".join(t[:6]))
    if out["top_unigrams"]:
        topu = ", ".join([f"{w}({c})" for w,c in out["top_unigrams"][:8]])
        lines.append("Top terms: " + topu)
    out["local_summary"] = "\n".join(lines)
    return out

# -------------------------
# Plot helpers (matplotlib)
# -------------------------
def plot_timeline(timeline_series, title="Events over time", figsize=(8,3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(timeline_series.index, timeline_series.values)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    fig.autofmt_xdate()
    return fig

def plot_top_terms(top_terms, title="Top terms", figsize=(6,3)):
    terms = [t for t,c in top_terms]
    counts = [c for t,c in top_terms]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(terms)), counts)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=45, ha='right')
    ax.set_title(title)
    fig.tight_layout()
    return fig

# -------------------------
# Abstract generation helpers
# -------------------------
def local_abstract_from_results(correlation_results: List[Dict[str,Any]]) -> str:
    if not correlation_results:
        return "No correlation results to summarize."
    total = len(correlation_results)
    matched = sum(1 for r in correlation_results if r.get("matches"))
    lines = []
    lines.append(f"Executive summary: {total} anomalies analyzed, {matched} had at least one matching log entry.")
    theme_counts = {}
    for r in correlation_results:
        sig = r.get("signature","")
        theme_counts[sig] = theme_counts.get(sig,0) + 1
    top_themes = sorted(theme_counts.items(), key=lambda x:-x[1])[:5]
    if top_themes:
        lines.append("Top anomaly signatures:")
        for s,c in top_themes:
            lines.append(f"- {s} ({c} occurrences)")
    lines.append("Anomaly hypotheses (short):")
    for r in correlation_results:
        ts = r.get("timestamp","?")
        sig = r.get("signature","")
        if r.get("matches"):
            top = r["matches"][0]
            snippet = (top.get("text","")[:120]).replace("\n"," ")
            lines.append(f"- {ts}: signature={sig} → likely related to log: \"{snippet}\" (score {top.get('score',0):.3f})")
        else:
            lines.append(f"- {ts}: signature={sig} → no log matches found in window.")
    lines.append("Recommended next steps:")
    lines.append("1. Inspect sensors in top signatures and cross-check maintenance logs.")
    lines.append("2. If repeated, schedule focused telemetry & manual inspection.")
    lines.append("3. Consider embedding-based log matching for deeper semantic matches.")
    return "\n".join(lines)

def llm_abstract_from_results(client, correlation_results: List[Dict[str,Any]]) -> str:
    compact = json.dumps(correlation_results, indent=2)[:16000]
    prompt = (
        "You are an expert industrial data scientist. Produce:\n"
        "1) EXECUTIVE SUMMARY (3–5 sentences)\n"
        "2) For each anomaly provide a 1-line root-cause hypothesis linking sensor signature to matched logs (if any)\n"
        "3) Prioritized next investigative steps (3 items)\n\n"
        "Input JSON:\n" + compact
    )
    resp = client([HumanMessage(content=prompt)])
    out = getattr(resp, "content", None)
    if out is None:
        return str(resp)
    return out

def make_groq_client(groq_key: str, model: str="llama3-70b-8192"):
    if not _HAS_GROQ:
        raise RuntimeError("langchain_groq not installed.")
    return ChatGroq(groq_api_key=groq_key, model=model)

# -------------------------
# File generation helpers & downloads
# -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode("utf-8")

def json_to_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, indent=2, default=str)
    return s.encode("utf-8")

def text_to_bytes(s: str) -> bytes:
    return s.encode("utf-8")

def correlation_to_dataframe(corr_results: List[Dict[str,Any]]) -> pd.DataFrame:
    rows = []
    for r in corr_results:
        base = {
            "anomaly_pos": r.get("anomaly_pos"),
            "anomaly_timestamp": r.get("timestamp"),
            "signature": r.get("signature")
        }
        matches = r.get("matches", [])
        if not matches:
            rows.append({**base, **{"match_idx": None, "match_timestamp": None, "match_score": None, "match_text": None}})
        else:
            for i, m in enumerate(matches):
                rows.append({**base, **{
                    "match_idx": i,
                    "match_timestamp": m.get("timestamp"),
                    "match_score": m.get("score"),
                    "match_text": m.get("text")
                }})
    return pd.DataFrame(rows)

def make_1month_sensor_and_logs(start_ts="2025-01-01 00:00:00", days=30,
                                out_sensor_path="/mnt/data/sensor_pressure_1month_minute.csv",
                                out_logs_path="/mnt/data/operator_logs_formatted_1month.csv",
                                seed=123):
    periods = days * 24 * 60
    rng = pd.date_range(start=start_ts, periods=periods, freq="T")
    np.random.seed(seed)
    pressure = 101.5 + 0.05 * np.sin(np.linspace(0, 20 * np.pi, periods)) + np.random.normal(scale=0.05, size=periods)
    temperature = 78.2 + 0.02 * np.sin(np.linspace(0, 10 * np.pi, periods)) + np.random.normal(scale=0.03, size=periods)
    vibration = 0.12 + 0.005 * np.sin(np.linspace(0, 8 * np.pi, periods)) + np.random.normal(scale=0.005, size=periods)
    anomaly_indices = [10, 11, 670, 671, 3000, 20000]
    for i in anomaly_indices:
        if i < periods:
            pressure[i:i+3] += np.random.uniform(8, 12)
            temperature[i:i+2] += np.random.uniform(10, 14)
            vibration[i:i+4] += np.random.uniform(0.4, 0.7)
    sensor_df = pd.DataFrame({
        "timestamp": rng,
        "pressure": np.round(pressure, 3),
        "temperature": np.round(temperature, 3),
        "vibration": np.round(vibration, 3)
    })
    log_entries = []
    for i in anomaly_indices:
        if i < periods:
            ts1 = rng[i] + pd.Timedelta(seconds=30)
            ts2 = rng[i] + pd.Timedelta(seconds=90)
            log_entries.append((ts1, "Minor vibration noise detected near pump A."))
            log_entries.append((ts2, "Pressure spike observed; operators alerted."))
    for h in range(0, periods, 60*12):
        ts = rng[h] + pd.Timedelta(seconds=10)
        log_entries.append((ts, "Routine check completed. All sensors nominal."))
    trend_times = [rng[9] + pd.Timedelta(seconds=45), rng[600] + pd.Timedelta(seconds=20)]
    for t in trend_times:
        log_entries.append((t, "Noticed rising temperature trend in the section."))
    log_entries_sorted = sorted(log_entries, key=lambda x: x[0])
    logs_df = pd.DataFrame(log_entries_sorted, columns=["timestamp", "log"])
    sensor_df.to_csv(out_sensor_path, index=False)
    logs_df.to_csv(out_logs_path, index=False)
    return sensor_df, logs_df, (out_sensor_path, out_logs_path)

# -------------------------
# Highlight helper
# -------------------------
def highlight_anomalies(df: pd.DataFrame, anomaly_series: pd.Series):
    try:
        mask = anomaly_series.reindex(df.index).fillna(False).astype(bool)
    except Exception:
        mask = pd.Series(False, index=df.index)
    def _row_style(row):
        try:
            if mask.loc[row.name]:
                return ['background-color: #ffcccc'] * len(row)
        except Exception:
            pass
        return [''] * len(row)
    return df.style.apply(_row_style, axis=1)

# -------------------------
# LLM prompt builder (context-aware)
# -------------------------
def build_insightful_chat_prompt(
    user_msg: str,
    sensor_df=None,
    logs_df=None,
    correlation_results=None,
    log_analysis=None,
) -> str:
    context_parts = []
    if sensor_df is not None:
        try:
            context_parts.append(
                f"Sensor dataframe shape: {sensor_df.shape}, columns: {list(sensor_df.columns)}"
            )
            try:
                sample_head = sensor_df.head(3).reset_index().to_dict(orient="records")
                context_parts.append("Sensor sample rows: " + json.dumps(sample_head, default=str))
            except Exception:
                pass
        except Exception:
            pass
    if log_analysis is not None:
        try:
            summary = log_analysis.get("local_summary", "")
            sev = log_analysis.get("severity_counts", {})
            top_terms = log_analysis.get("top_unigrams", [])[:8]
            context_parts.append("Log analysis summary:")
            if summary:
                context_parts.append(summary)
            if sev:
                context_parts.append(f"Log severity counts: {sev}")
            if top_terms:
                context_parts.append("Top log terms: " + ", ".join([f"{w}({c})" for w, c in top_terms]))
        except Exception:
            pass
    if correlation_results:
        try:
            sample_corr = correlation_results[:5]
            corr_json = json.dumps(sample_corr, indent=2, default=str)
            corr_json = corr_json[:4000]
            context_parts.append("Sample anomaly–log correlations (truncated JSON):\n" + corr_json)
        except Exception:
            pass
    if not context_parts:
        context_text = "No sensor or log data is loaded in the dashboard. Answer the user's question generally."
    else:
        context_text = "\n\n".join(context_parts)
    prompt = (
        "You are an assistant embedded in a Streamlit dashboard for industrial sensor anomaly detection.\n"
        "Use the CONTEXT to answer the user's question succinctly and concretely. Cite sample values or timestamps if helpful.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION:\n{user_msg}"
    )
    return prompt

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Anomaly+Logs+Chat", layout="wide")
st.title("Sensor Anomaly Detection + Log Correlation + Context Chat")

# Sidebar: Groq key and controls
st.sidebar.header("LLM (Optional) - Groq")
groq_key_input = st.sidebar.text_input("Paste Groq API Key (optional)", type="password")
init_llm = st.sidebar.button("Initialize Groq Client")

if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
    st.session_state.groq_ready = False

if init_llm:
    if not groq_key_input:
        st.sidebar.error("Paste Groq API key first.")
    else:
        try:
            client = make_groq_client(groq_key_input)
            st.session_state.groq_client = client
            st.session_state.groq_ready = True
            st.sidebar.success("Groq client ready.")
        except Exception as e:
            st.session_state.groq_client = None
            st.session_state.groq_ready = False
            st.sidebar.error(f"Failed to init Groq client: {e}")

# Anomaly options
st.sidebar.header("Anomaly detection options")
method = st.sidebar.selectbox("Method", [
    "isolation_forest","zscore","ocsvm","lof","elliptic","pca","iqr","lstm_autoencoder"
])
contamination = st.sidebar.slider("IsolationForest contamination", 0.001, 0.2, 0.01, step=0.001)
z_thresh = st.sidebar.slider("Z-score threshold", 2.0, 6.0, 3.5, step=0.1)
z_window = st.sidebar.number_input("Z-score rolling window (0=global)", min_value=0, value=0, step=1)
iqr_window = st.sidebar.number_input("IQR rolling window rows", min_value=5, value=48, step=1)
lstm_seq = st.sidebar.number_input("LSTM seq length", min_value=5, value=24, step=1)
lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, value=3, step=1)

# Layout: left = main, right = chat & small controls
left, right = st.columns([3,1])
with left:
    st.header("Upload / Generate Data")
    sensor_file = st.file_uploader("Sensor CSV (timestamp optional)", type=["csv"])
    logs_file = st.file_uploader("Operator logs CSV (must include 'log' and optional 'timestamp')", type=["csv"])

    if st.button("Generate 1-month minute-level synthetic data (and save to /mnt/data)"):
        sensor_df_gen, logs_df_gen, paths = make_1month_sensor_and_logs()
        st.success(f"Generated files: {paths[0]}, {paths[1]}")
        with open(paths[0], "rb") as f:
            st.download_button("Download generated sensor CSV", data=f, file_name=os.path.basename(paths[0]), mime="text/csv")
        with open(paths[1], "rb") as f:
            st.download_button("Download generated logs CSV", data=f, file_name=os.path.basename(paths[1]), mime="text/csv")

    # Load uploaded or generated examples if user didn't upload
    sensor_df = None
    logs_df = None
    if sensor_file:
        try:
            sensor_df = pd.read_csv(sensor_file)
            if "timestamp" in sensor_df.columns:
                try:
                    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
                    sensor_df = sensor_df.set_index("timestamp").sort_index()
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Failed reading sensor CSV: {e}")
            sensor_df = None
    else:
        example_path = "/mnt/data/sensor_pressure_1month_minute.csv"
        if os.path.exists(example_path):
            try:
                tmp = pd.read_csv(example_path)
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
                sensor_df = tmp.set_index("timestamp").sort_index()
            except Exception:
                sensor_df = None

    if logs_file:
        try:
            logs_df = pd.read_csv(logs_file)
        except Exception as e:
            st.error(f"Failed reading logs CSV: {e}")
            logs_df = None
    else:
        example_logs_path = "/mnt/data/operator_logs_formatted_1month.csv"
        if os.path.exists(example_logs_path):
            try:
                logs_df = pd.read_csv(example_logs_path)
            except Exception:
                logs_df = None

    # Run anomaly detection if sensor data loaded
    correlation_results: List[Dict[str,Any]] = []
    anomaly_series = None
    log_analysis = None

    if sensor_df is not None:
        st.subheader("Sensor preview")
        st.dataframe(sensor_df.head())

        try:
            if method == "isolation_forest":
                anomaly_series = detect_anomalies_isolation(sensor_df, contamination=contamination)
            elif method == "zscore":
                w = int(z_window) if z_window > 0 else None
                anomaly_series = detect_anomalies_zscore(sensor_df, z_thresh=z_thresh, window=w)
            elif method == "ocsvm":
                anomaly_series = detect_anomalies_ocsvm(sensor_df)
            elif method == "lof":
                anomaly_series = detect_anomalies_lof(sensor_df)
            elif method == "elliptic":
                anomaly_series = detect_anomalies_elliptic(sensor_df)
            elif method == "pca":
                anomaly_series = detect_anomalies_pca(sensor_df)
            elif method == "iqr":
                anomaly_series = detect_anomalies_iqr(sensor_df, window=int(iqr_window))
            elif method == "lstm_autoencoder":
                st.warning("Training LSTM Autoencoder (may be slow).")
                anomaly_series = detect_anomalies_lstm_autoencoder(sensor_df, seq_len=int(lstm_seq), epochs=int(lstm_epochs))
            else:
                anomaly_series = pd.Series([False]*len(sensor_df), index=sensor_df.index)
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
            anomaly_series = pd.Series([False]*len(sensor_df), index=sensor_df.index)

        st.subheader("Anomalies detected")
        st.write("Count:", int(anomaly_series.sum()))

        try:
            styled = highlight_anomalies(sensor_df, anomaly_series)
            st.subheader("Sensor table (anomalies highlighted in red)")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.write("Could not render styled table — showing anomalous rows only.")
            try:
                st.dataframe(sensor_df[anomaly_series])
            except Exception:
                st.write(list(np.where(anomaly_series)[0]))

        if logs_df is not None:
            try:
                correlation_results = correlate_anomalies_with_logs(sensor_df, logs_df, anomaly_series, window_minutes=30, top_matches=3)
            except Exception as e:
                st.error(f"Correlation failed: {e}")
                correlation_results = []

            st.subheader("Anomaly → Top matching logs")
            if not correlation_results:
                st.write("No results (no logs or no anomalies).")
            else:
                for r in correlation_results:
                    st.markdown(f"**Anomaly (pos={r['anomaly_pos']}, ts={r['timestamp']})** — signature: *{r['signature']}*")
                    if not r["matches"]:
                        st.write("No matching logs found in window.")
                    else:
                        for m in r["matches"]:
                            st.write(f"- score {m['score']:.3f}: {m['text']}")
                    st.markdown("---")
        else:
            st.info("Upload operator logs CSV to correlate anomalies and produce abstracts.")

        # Abstract generation (local)
        st.subheader("Abstract / Executive Summary")
        if st.button("Generate Abstract Summary (local)"):
            st.write(local_abstract_from_results(correlation_results))

        # Downloads
        st.subheader("Download results")
        if sensor_df is not None:
            st.download_button("Download sensor data (CSV)", data=df_to_csv_bytes(sensor_df.reset_index()), file_name="sensor_data_download.csv", mime="text/csv")
        if anomaly_series is not None and sensor_df is not None:
            try:
                anom_df = sensor_df[anomaly_series].reset_index()
                st.download_button("Download detected anomalies (CSV)", data=df_to_csv_bytes(anom_df), file_name="detected_anomalies.csv", mime="text/csv")
            except Exception:
                pass
        if correlation_results:
            st.download_button("Download correlation results (JSON)", data=json_to_bytes(correlation_results), file_name="correlation_results.json", mime="application/json")
            corr_df = correlation_to_dataframe(correlation_results)
            st.download_button("Download correlation results (CSV)", data=df_to_csv_bytes(corr_df), file_name="correlation_results.csv", mime="text/csv")

    else:
        st.info("Upload or generate sensor data to start analysis.")

    # Log analysis panel
    if logs_df is not None:
        st.subheader("Log Analysis")
        try:
            log_analysis = analyze_logs(logs_df, text_col="log")
            st.write("Severity counts:", log_analysis["severity_counts"])
            st.write("Top unigrams:", log_analysis["top_unigrams"][:12])
            st.write("Top bigrams:", log_analysis["top_bigrams"][:12])
            st.text("Local summary:")
            st.text(log_analysis["local_summary"])
            if not log_analysis["timeline"].empty:
                fig = plot_timeline(log_analysis["timeline"], title="Log events over time")
                st.pyplot(fig)
            if log_analysis["top_unigrams"]:
                fig2 = plot_top_terms(log_analysis["top_unigrams"][:12], title="Top unigrams")
                st.pyplot(fig2)
            if log_analysis.get("cluster_representatives") is not None and log_analysis.get("df_with_index") is not None:
                st.markdown("**Cluster representatives:**")
                reps = log_analysis["cluster_representatives"]
                df_idxed = log_analysis["df_with_index"]
                for c, idx in reps.items():
                    st.markdown(f"- Cluster {c} example (row {idx}):")
                    try:
                        st.write(df_idxed.iloc[idx]["log"])
                    except Exception:
                        pass
            # downloads for log analysis
            st.download_button("Download full log analysis (JSON)", data=json_to_bytes(log_analysis), file_name="log_analysis.json", mime="application/json")
            if "timeline" in log_analysis and not log_analysis["timeline"].empty:
                ts_df = log_analysis["timeline"].reset_index()
                ts_df.columns = ["period", "count"]
                st.download_button("Download log timeline (CSV)", data=df_to_csv_bytes(ts_df), file_name="log_timeline.csv", mime="text/csv")
            if "most_common_cleaned" in log_analysis:
                mc = pd.DataFrame(log_analysis["most_common_cleaned"], columns=["cleaned_text","count"])
                st.download_button("Download most common cleaned log phrases (CSV)", data=df_to_csv_bytes(mc), file_name="log_common_phrases.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Log analysis failed: {e}")

with right:
    st.header("Chat (Direct Groq)")
    if not _HAS_GROQ:
        st.info("Install 'langchain_groq' and 'langchain' to enable LLM chat.")
    else:
        if st.session_state.get("groq_ready"):
            st.success("Groq client initialized (chat ready).")
        else:
            st.info("Paste Groq key in sidebar and click Initialize Groq Client.")

    user_msg = st.text_input("Chat message")
    if st.button("Send Chat Message"):
        if not _HAS_GROQ:
            st.error("Chat unavailable — 'langchain_groq' not installed.")
        elif not st.session_state.get("groq_ready") or not st.session_state.get("groq_client"):
            st.error("Groq client not initialized. Use sidebar.")
        else:
            try:
                client = st.session_state.groq_client
                prompt = build_insightful_chat_prompt(
                    user_msg=user_msg,
                    sensor_df=sensor_df,
                    logs_df=logs_df,
                    correlation_results=correlation_results,
                    log_analysis=log_analysis,
                )
                resp = client([HumanMessage(content=prompt)])
                out = getattr(resp, "content", None)
                if out is None:
                    st.write(str(resp))
                else:
                    st.markdown("**Assistant:**")
                    st.write(out)
            except Exception as e:
                st.error(f"Chat failed: {e}")

st.markdown("---")
st.caption("Notes: For better semantic log matching, replace TF-IDF with embeddings. LSTM autoencoder requires Keras/TensorFlow; if unavailable, use other detectors.")
