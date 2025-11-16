# generate_data.py
"""
Generate synthetic sensor + operator logs CSV files.

Outputs:
- /mnt/data/sensor_pressure_1month_minute.csv
- /mnt/data/operator_logs_formatted_1month.csv

Requirements:
    pip install pandas numpy
"""

import numpy as np
import pandas as pd
from datetime import timedelta

def generate_sensor_minute_data(start_ts="2025-01-01 00:00:00",
                                days=30,
                                out_path="/mnt/data/sensor_pressure_1month_minute.csv",
                                seed=123,
                                anomaly_specs=None):
    """
    Generate minute-level sensor data for `days` days.
    Columns: timestamp, pressure, temperature, vibration
    anomaly_specs: list of dicts with keys:
        - 'index' (int) OR 'timestamp' (pd.Timestamp)
        - 'dur' (int minutes)
        - 'pressure_delta', 'temp_delta', 'vib_delta' (floats or callables)
    """
    periods = days * 24 * 60  # minutes
    rng = pd.date_range(start=start_ts, periods=periods, freq="T")

    np.random.seed(seed)

    # Baseline signals with small periodic components + noise
    pressure = 101.5 + 0.05 * np.sin(np.linspace(0, 20 * np.pi, periods)) + np.random.normal(scale=0.05, size=periods)
    temperature = 78.2 + 0.02 * np.sin(np.linspace(0, 10 * np.pi, periods)) + np.random.normal(scale=0.03, size=periods)
    vibration = 0.12 + 0.005 * np.sin(np.linspace(0, 8 * np.pi, periods)) + np.random.normal(scale=0.005, size=periods)

    # default anomaly specs if none provided
    if anomaly_specs is None:
        anomaly_specs = [
            {"index": 10, "dur": 3, "pressure_delta": 9.0, "temp_delta": 12.0, "vib_delta": 0.5},
            {"index": 670, "dur": 4, "pressure_delta": 10.0, "temp_delta": 11.0, "vib_delta": 0.45},
            {"index": 3000, "dur": 5, "pressure_delta": 8.5, "temp_delta": 13.0, "vib_delta": 0.6},
            {"index": 20000, "dur": 6, "pressure_delta": 9.5, "temp_delta": 10.0, "vib_delta": 0.55},
        ]

    # Apply anomalies
    for spec in anomaly_specs:
        # resolve start index
        if "index" in spec:
            start_idx = int(spec["index"])
        elif "timestamp" in spec:
            # find nearest index for timestamp
            start_idx = int((pd.to_datetime(spec["timestamp"]) - pd.to_datetime(start_ts)).total_seconds() // 60)
        else:
            continue

        dur = int(spec.get("dur", 1))
        p_delta = spec.get("pressure_delta", 0.0)
        t_delta = spec.get("temp_delta", 0.0)
        v_delta = spec.get("vib_delta", 0.0)

        # clip to array bounds
        s = max(0, start_idx)
        e = min(periods, start_idx + dur)
        if s >= e:
            continue

        # If deltas are callable, call per index
        if callable(p_delta):
            pressure[s:e] += p_delta(np.arange(e - s))
        else:
            pressure[s:e] += p_delta

        if callable(t_delta):
            temperature[s:e] += t_delta(np.arange(e - s))
        else:
            temperature[s:e] += t_delta

        if callable(v_delta):
            vibration[s:e] += v_delta(np.arange(e - s))
        else:
            vibration[s:e] += v_delta

    # Round reasonably
    pressure = np.round(pressure, 3)
    temperature = np.round(temperature, 3)
    vibration = np.round(vibration, 3)

    df = pd.DataFrame({
        "timestamp": rng,
        "pressure": pressure,
        "temperature": temperature,
        "vibration": vibration
    })

    df.to_csv(out_path, index=False)
    return df, out_path

def generate_logs_from_sensor(sensor_df,
                              out_path="operator_logs_formatted_1month.csv",
                              around_minutes=2,
                              extra_routine_every_minutes=12*60):
    """
    Create log messages at times near large deviations (simple heuristic),
    plus routine checks every `extra_routine_every_minutes`.
    Returns logs_df, path.
    """
    # detect candidate anomaly points by looking for large jumps in pressure or vibration
    # compute short-window diff magnitude
    df = sensor_df.copy()
    df["pressure_diff"] = df["pressure"].diff().abs().fillna(0)
    df["temp_diff"] = df["temperature"].diff().abs().fillna(0)
    df["vib_diff"] = df["vibration"].diff().abs().fillna(0)

    # threshold heuristics
    p_thresh = max(0.5, df["pressure_diff"].mean() + 3 * df["pressure_diff"].std())
    t_thresh = max(0.5, df["temp_diff"].mean() + 3 * df["temp_diff"].std())
    v_thresh = max(0.05, df["vib_diff"].mean() + 3 * df["vib_diff"].std())

    candidate_idxs = set()
    for i, row in df.iterrows():
        if row["pressure_diff"] >= p_thresh or row["temp_diff"] >= t_thresh or row["vib_diff"] >= v_thresh:
            candidate_idxs.add(i)

    # Build log entries near candidate_idxs
    logs = []
    for idx in sorted(candidate_idxs):
        ts = pd.to_datetime(df.loc[idx, "timestamp"])
        # create one or two log messages around the minute
        logs.append((ts + pd.Timedelta(seconds=30), "Minor vibration noise detected near pump A."))
        logs.append((ts + pd.Timedelta(seconds=90), "Pressure spike observed; operators alerted."))

    # Routine entries
    start = pd.to_datetime(df["timestamp"].iloc[0])
    end = pd.to_datetime(df["timestamp"].iloc[-1])
    routine_ts = pd.date_range(start=start, end=end, freq=f"{extra_routine_every_minutes}T")
    for t in routine_ts:
        logs.append((t + pd.Timedelta(seconds=10), "Routine check completed. All sensors nominal."))

    # Some trend notices (simple)
    # e.g., if temperature rolling mean increases significantly over an hour
    temp_roll = df["temperature"].rolling(window=60, min_periods=1).mean()
    for i in range(1, len(temp_roll)):
        if temp_roll.iloc[i] - temp_roll.iloc[i-1] > 0.5:
            t = pd.to_datetime(df.loc[i, "timestamp"]) + pd.Timedelta(seconds=45)
            logs.append((t, "Noticed rising temperature trend in the section."))

    # Consolidate, sort, deduplicate timestamps (if necessary)
    logs_sorted = sorted(logs, key=lambda x: x[0])
    logs_df = pd.DataFrame(logs_sorted, columns=["timestamp", "log"])

    # drop near-duplicates (same timestamp + text)
    logs_df = logs_df.drop_duplicates(subset=["timestamp", "log"]).reset_index(drop=True)

    logs_df.to_csv(out_path, index=False)
    return logs_df, out_path

def main():
    # Example usage: generate 1 month minute-level files
    sensor_df, sensor_path = generate_sensor_minute_data(
        start_ts="2025-01-01 00:00:00",
        days=30,
        out_path="sensor_pressure_1month_minute.csv",
        seed=123,
        anomaly_specs=[
            {"index": 10, "dur": 3, "pressure_delta": 9.0, "temp_delta": 12.0, "vib_delta": 0.5},
            {"index": 670, "dur": 4, "pressure_delta": 10.0, "temp_delta": 11.0, "vib_delta": 0.45},
            {"index": 3000, "dur": 5, "pressure_delta": 8.5, "temp_delta": 13.0, "vib_delta": 0.6},
            {"index": 20000, "dur": 6, "pressure_delta": 9.5, "temp_delta": 10.0, "vib_delta": 0.55},
        ]
    )
    print(f"Sensor file written to: {sensor_path} ({len(sensor_df)} rows)")

    logs_df, logs_path = generate_logs_from_sensor(sensor_df,
                                                   out_path="operator_logs_formatted_1month.csv",
                                                   around_minutes=2,
                                                   extra_routine_every_minutes=12*60)
    print(f"Logs file written to: {logs_path} ({len(logs_df)} rows)")

if __name__ == "__main__":
    main()
