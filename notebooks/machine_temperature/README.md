# Machine Temperature Dataset

Notebooks analyzing the NAB machine_temperature_system_failure time series (univariate, labeled anomaly intervals). The data captures temperature readings from an industrial machine leading up to a known failure.

## Notebooks Included

- `01_exploration.ipynb`  
Visualizes the raw temperature time series, inspects statistical properties and failure windows

- `02_statistical_baselines.ipynb`  
Implements statistical anomaly detection methods including Z-score, rolling mean, IQR, and MAD. Detected anomalies are compared directly against NAB-provided ground truth intervals.

- `03_deep_learning.ipynb`  
Trains an LSTM autoencoder on normal segments of the data. Reconstruction error is used to detect deviations indicating potential anomalies.


## Objectives

- Identify early warning signs of failure
- Compare performance of statistical vs. deep learning methods for anomaly detection
- Visualize and benchmark detection performance against labeled anomalies from NAB

## Dataset Info

- Source: NAB `realKnownCause/machine_temperature_system_failure.csv` from the NAB GitHub Repository: https://github.com/numenta/NAB
- Features:  
  - `timestamp` – datetime  
  - `value` – temperature reading
- Ground truth anomaly labels from NAB are loaded from ../labels/combined_windows.json for validation and scoring.



- Full dataset available via [NAB GitHub](https://github.com/numenta/NAB)


