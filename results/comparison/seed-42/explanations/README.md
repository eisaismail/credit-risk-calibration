# Prediction explanations

Explained 6,000 real held-out records using Tree SHAP.
Positive values raise the model's default score; negative values lower it. Values use log-odds, not percentage points.
These describe model behaviour and do not show why a person actually defaulted.

| Real data row | Observed default | Raw probability | Calibrated probability | Largest contribution |
|---|---:|---:|---:|---|
| 16866 | 0 | 0.351 | 0.157 | PAY_0 = -2: lowers model risk score |
| 26727 | 0 | 0.194 | 0.082 | PAY_0 = 0: lowers model risk score |
| 22272 | 0 | 0.175 | 0.074 | PAY_0 = 0: lowers model risk score |
| 14422 | 0 | 0.318 | 0.140 | PAY_0 = 0: lowers model risk score |
| 16233 | 0 | 0.111 | 0.047 | PAY_0 = 0: lowers model risk score |

All 23 contributions plus the base score reconstruct each uncalibrated score; the chart displays only the top eight.
Maximum score reconstruction error across all rows: 1.15e-14.
The calibrated probability is shown separately; these SHAP values do not explain the calibration layer.
Rows were sampled with a fixed seed, not selected for correct predictions. See JSON for all contributions and observed values.
