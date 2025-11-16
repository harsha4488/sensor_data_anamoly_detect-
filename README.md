# sensor_data_anamoly_detect

Sensor Anomaly Detection & Log Correlation Prototype
----------------------------------------------------

This project provides a Streamlit-based application for detecting anomalies in
sensor time-series data and correlating them with operator log messages.

Main Features:
- Multiple anomaly detection methods (Isolation Forest, Z-Score, LOF, PCA, etc.)
- Optional LSTM Autoencoder for deep learning–based anomaly detection
- Log correlation using TF-IDF similarity
- Log analysis (top terms, topics, clusters, severity counts)
- Optional Groq LLM support for summaries and explanations
- Downloadable results (sensor data, anomalies, correlation outputs, abstracts)
- Synthetic data generator (30-day minute-level data)

Files:
- app.py .................. Main Streamlit app
- insights_nlp.py ........ A Streamlit-based system that detects sensor anomalies and correlates them with operator logs.

- generate_data.py ........ Script to generate synthetic sensor + log CSV files
- sensor_pressure_1month_minute.csv .... Example generated sensor data
- operator_logs_formatted_1month.csv .... Example generated log data
- requirements.txt ........ Package dependencies

Run the application:
    streamlit run app.py

Generate new synthetic data:
    python generate_data.py

This prototype is intended for demonstration and experimentation with
industrial anomaly detection and text-log correlation workflows.
