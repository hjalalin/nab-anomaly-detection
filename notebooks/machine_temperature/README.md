# Machine Temperature Dataset

This folder contains notebooks analyzing the `machine_temperature_system_failure.csv` dataset from the NAB (Numenta Anomaly Benchmark).
The dataset captures temperature readings of an industrial machine leading up to a known system failure. 

## Notebooks Included

- `01_exploration.ipynb`  
  Basic data inspection and time series plots.

- `02_statistical_methods.ipynb`  
  Z-score thresholding and rolling mean-based anomaly detection.

- `03_lstm_autoencoder.ipynb`  
  LSTM autoencoder trained on normal data to detect anomalies via reconstruction error.

- `04_trend_detection.ipynb`  
  Detects gradual upward trends using rolling slopes and statistical trend tests (e.g., Mann-Kendall).

## Objectives

- Identify early warning signs of failure
- Compare performance of detection techniques across statistical and deep learning approaches

## Dataset Info

- Source: NAB `realKnownCause/machine_temperature_system_failure.csv`
- Features:  
  - `timestamp` – datetime  
  - `value` – temperature reading



- A smaller version of the dataset (`machine_temperature_small.csv`) is used to keep the repo lightweight.
- Full dataset available via [NAB GitHub](https://github.com/numenta/NAB)

---

> These notebooks demonstrate a complete workflow from raw data to anomaly detection using interpretable and scalable methods.
