# Machine Temperature Dataset

This folder contains notebooks analyzing the `machine_temperature_system_failure.csv` dataset from the NAB (Numenta Anomaly Benchmark). This is a univariate time-series with labeled anomalies.
The dataset captures temperature readings of an industrial machine leading up to a known system failure. 

## Notebooks Included

- `01_exploration.ipynb`  
Visualizes the raw temperature time series, inspects statistical properties

- `02_statistical_methods.ipynb`  
Implements statistical anomaly detection methods including Z-score, rolling mean, IQR, and MAD. Detected anomalies are compared directly against NAB-provided ground truth intervals.

- `03_lstm_autoencoder.ipynb`  
Trains an LSTM autoencoder on normal segments of the data. Reconstruction error is used to detect deviations indicating potential anomalies.

- `04_trend_detection.ipynb`  
Focuses on identifying gradual upward trends using rolling linear regression and non-parametric statistical tests like the Mann-Kendall trend test.

## Objectives

- Identify early warning signs of failure
- Compare performance of atistical vs. deep learning methods for anomaly detection
- Visualize and benchmark detection performance against labeled anomalies from NAB

## Dataset Info

- Source: NAB `realKnownCause/machine_temperature_system_failure.csv` from the NAB GitHub Repository: https://github.com/numenta/NAB
- Features:  
  - `timestamp` – datetime  
  - `value` – temperature reading
- Ground truth anomaly labels from NAB are loaded from ../labels/combined_windows.json for validation and scoring.



- A smaller version of the dataset (`machine_temperature_small.csv`) is used to keep the repo lightweight.
- Full dataset available via [NAB GitHub](https://github.com/numenta/NAB)

---

> These notebooks demonstrate a complete workflow from raw data to anomaly detection using interpretable and scalable methods.
