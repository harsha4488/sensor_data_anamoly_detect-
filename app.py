# app.py
"""
Streamlit app (single-file):
- Multi-model anomaly detection (IsolationForest, Z-score, OCSVM, LOF, Elliptic, PCA, IQR, LSTM Autoencoder)
- Robust TF-IDF log correlation
- Lightweight log analysis toolkit (timeline, top ngrams, topics via NMF, clustering)
- Direct Groq LLM chat (optional — paste key in sidebar to enable)
- Generate abstract summary (LLM if initialized, otherwise local fallback)
- Highlight anomaly rows in sensor table (red)
Notes:
- LSTM autoencoder requires Keras (tensorflow.keras or standalone keras). The app attempts both.
- If you don't want LSTM, you can ignore Keras / TensorFlow installation.
- To enable chat/LMM features, install 'langchain_groq' and 'langchain' and paste a valid GROQ key in the sidebar.
Run:
    streamlit run app.py
"""

import os
import re
import json
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# sklearn
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

# Keras import strategy (try tensorflow.keras then fallback to standalone keras)
_KERAS_IMPL = None
layers = None
models = None
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras import layers, models
    _KERAS_IMPL = "tf.keras"
except Exception:
    try:
        import keras  # noqa: F401
        from keras import layers, models  # type: ignore
        _KERAS_IMPL = "keras"
    except Exception:
        _KERAS_IMPL = None
        layers = None
        models = None

# -------------------------
# Utilities
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    toks = re.findall(r"[a-zA-Z]+", text)
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

# -------------------------
# LSTM autoencoder (if available)
# -------------------------
def build_lstm_autoencoder(seq_len:int, nfeat:int):
    if models is None or layers is None:
        raise RuntimeError("Keras is not available in this environment. Install tensorflow or keras.")
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
# Anomaly signature & robust correlation
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
    """
    Robust correlation between anomalies and logs.
    Uses index->position mapping to avoid get_loc returning slices.
    """
    results: List[Dict[str,Any]] = []

    logs = logs_df.copy()
    # try to set datetime index if possible
    if "timestamp" in logs.columns:
        try:
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])
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

    # Build TF-IDF matrix (if text exists)
    corpus_texts = logs["_preproc_text"].fillna("").tolist()
    tfidf = None
    logs_vec = None
    if any(corpus_texts):
        tfidf = TfidfVectorizer(max_features=2000)
        logs_vec = tfidf.fit_transform(corpus_texts)

    # index -> first integer position mapping
    index_to_pos = {}
    for p, idx in enumerate(logs.index):
        if idx not in index_to_pos:
            index_to_pos[idx] = p

    # anomaly integer positions
    anomaly_positions = np.where(anomaly_series)[0]
    sensor_idx = sensor_df.index if isinstance(sensor_df.index, pd.DatetimeIndex) else None

    for pos in anomaly_positions:
        # get timestamp for anomaly if available
        ts = None
        try:
            ts = sensor_df.index[pos] if sensor_idx is not None else None
        except Exception:
            ts = None

        signature = anomaly_signature(sensor_df, pos)
        sig_proc = " ".join(simple_tokenize(signature))

        # select logs within window
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

        # map window log indices to integer positions
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
                log_ts = str(logs.index[global_pos])
                original_text = logs["log"].iloc[global_pos]
                matches.append({
                    "log_pos": int(global_pos),
                    "log_index": str(logs.index[global_pos]),
                    "timestamp": log_ts,
                    "text": original_text,
                    "score": round(score, 4)
                })
        else:
            # fallback token overlap
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
    t = str(text)
    t = t.lower()
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
    counters = {}
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
        best = sims.argmax()
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
# Plot helpers
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

def make_groq_client(groq_key: str, model: str="openai/gpt-oss-20b"):
    if not _HAS_GROQ:
        raise RuntimeError("langchain_groq not installed. Install 'langchain_groq' and 'langchain' to use LLM features.")
    return ChatGroq(groq_api_key=groq_key, model=model)

# -------------------------
# Highlight helper
# -------------------------
def highlight_anomalies(df: pd.DataFrame, anomaly_series: pd.Series):
    try:
        mask = anomaly_series.reindex(df.index).fillna(False).astype(bool)
    except Exception:
        mask = pd.Series(False, index=df.index)
    def _row_style(row):
        if mask.loc[row.name]:
            return ['background-color: #ffcccc'] * len(row)
        else:
            return [''] * len(row)
    return df.style.apply(_row_style, axis=1)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Anomaly + Logs + Abstract", layout="wide")
st.title("Sensor Anomaly Detection + Log Correlation + Log Analysis + Abstract")

# Sidebar - Groq key / init
st.sidebar.header("Groq LLM (optional)")
groq_key_input = st.sidebar.text_input("Paste Groq API Key (optional, for LLM abstracts)", type="password")
init_llm = st.sidebar.button("Initialize Groq Client")

if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
    st.session_state.groq_ready = False

if init_llm:
    if not groq_key_input:
        st.sidebar.error("Paste Groq API key before initialization.")
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

# Sidebar - anomaly options
st.sidebar.header("Anomaly detection options")
method = st.sidebar.selectbox("Method", [
    "isolation_forest","zscore","ocsvm","lof","elliptic","pca","iqr","lstm_autoencoder"
])
contamination = st.sidebar.slider("IsolationForest contamination", 0.001, 0.2, 0.01, step=0.001)
z_thresh = st.sidebar.slider("Z-score threshold", 2.0, 6.0, 3.5, step=0.1)
z_window = st.sidebar.number_input("Z-score rolling window (0=global)", min_value=0, value=0, step=1)
iqr_window = st.sidebar.number_input("IQR rolling window rows", min_value=5, value=48, step=1)
lstm_seq = st.sidebar.number_input("LSTM seq length", min_value=5, value=24, step=1)
lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, value=5, step=1)

# Layout
left, right = st.columns([3,1])
with left:
    st.header("Upload data")
    sensor_file = st.file_uploader("Sensor CSV (must include numeric columns; optional 'timestamp')", type=["csv"])
    logs_file = st.file_uploader("Operator logs CSV (must contain 'log' and optional 'timestamp')", type=["csv"])

# State
correlation_results: List[Dict[str,Any]] = []
anomaly_series = None
sensor_df = None
logs_df = None
log_analysis = None

# Load uploaded files
if sensor_file:
    try:
        sensor_df = pd.read_csv(sensor_file)
        if "timestamp" in sensor_df.columns:
            try:
                sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
                sensor_df = sensor_df.set_index("timestamp").sort_index()
            except Exception:
                st.warning("Could not parse sensor timestamp column; using original index.")
    except Exception as e:
        st.error(f"Failed reading sensor CSV: {e}")
        sensor_df = None

if logs_file:
    try:
        logs_df = pd.read_csv(logs_file)
    except Exception as e:
        st.error(f"Failed reading logs CSV: {e}")
        logs_df = None

# Run detection & correlation
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

    # Highlighted table
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

    # Correlate with logs if present
    if logs_df is not None:
        try:
            correlation_results = correlate_anomalies_with_logs(sensor_df, logs_df, anomaly_series,
                                                                window_minutes=30, top_matches=3)
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

        # Abstract generation
        st.subheader("Abstract / Executive Summary")
        if st.button("Generate Abstract Summary (LLM if initialized, otherwise local)"):
            if st.session_state.get("groq_ready") and st.session_state.get("groq_client"):
                try:
                    client = st.session_state.groq_client
                    llm_text = llm_abstract_from_results(client, correlation_results)
                    st.markdown("**LLM-generated abstract:**")
                    st.write(llm_text)
                except Exception as e:
                    st.error(f"LLM abstract failed: {e}")
                    st.markdown("**Fallback local abstract:**")
                    st.write(local_abstract_from_results(correlation_results))
            else:
                st.info("Groq client not initialized — using local summary.")
                st.write(local_abstract_from_results(correlation_results))
    else:
        st.info("Upload operator logs CSV to correlate anomalies and produce abstracts.")

    # Log analysis panel (if logs uploaded)
    if logs_df is not None:
        st.subheader("Log Analysis")
        try:
            log_analysis = analyze_logs(logs_df, text_col="log")
            st.write("Severity counts:", log_analysis["severity_counts"])
            st.write("Top unigrams:", log_analysis["top_unigrams"][:12])
            st.write("Top bigrams:", log_analysis["top_bigrams"][:12])
            st.text("Local summary:")
            st.text(log_analysis["local_summary"])

            # Timeline plot
            if not log_analysis["timeline"].empty:
                fig = plot_timeline(log_analysis["timeline"], title="Log events over time")
                st.pyplot(fig)

            # Top terms plot
            if log_analysis["top_unigrams"]:
                fig2 = plot_top_terms(log_analysis["top_unigrams"][:12], title="Top unigrams")
                st.pyplot(fig2)

            # Show topical clusters examples
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
        except Exception as e:
            st.error(f"Log analysis failed: {e}")

else:
    st.info("Upload a sensor CSV to start analysis.")

# Chat pane (right column)
with right:
    st.header("Chat (Direct Groq)")
    if not _HAS_GROQ:
        st.info("Install 'langchain_groq' and 'langchain' to enable LLM chat (optional).")
    else:
        if st.session_state.get("groq_ready"):
            st.success("Groq client initialized (chat ready).")
        else:
            st.info("Paste Groq key in sidebar and click 'Initialize Groq Client' to enable chat.")

    user_msg = st.text_input("Chat message")
    if st.button("Send Chat Message"):
        if not _HAS_GROQ:
            st.error("Chat unavailable — 'langchain_groq' not installed.")
        elif not st.session_state.get("groq_ready") or not st.session_state.get("groq_client"):
            st.error("Groq client not initialized. Use sidebar.")
        else:
            try:
                client = st.session_state.groq_client
                response = llm_abstract_from_results(client, [{"timestamp":"chat","signature":"user_query","matches":[{"text":user_msg,"score":1.0}]}])
                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"Chat failed: {e}")

st.markdown("---")
st.caption(
    "Notes: For better semantic log matching, replace TF-IDF with embedding-based similarity (sentence-transformers). "
    "LSTM autoencoder requires Keras/TensorFlow; if unavailable, use other detectors."
)
