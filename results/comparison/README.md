# Real-data model comparison

UCI Default of Credit Card Clients: 30,000 actual historical records. Seeds: [42, 43, 44].
Each run uses 18,000 training, 6,000 calibration and 6,000 test rows. Models share the same split within each seed.
All settings were fixed before this comparison; the test set is not used for training, calibration or tuning.

| Model | ROC-AUC mean (range) | Brier mean | Log loss mean | ECE mean |
|---|---:|---:|---:|---:|
| Gradient boosting / calibrated (sigmoid) | 0.7782 (0.7737–0.7851) | 0.1351 | 0.4314 | 0.0123 |
| Gradient boosting / uncalibrated | 0.7782 (0.7737–0.7851) | 0.1745 | 0.5312 | 0.1834 |
| Logistic regression / calibrated (sigmoid) | 0.7230 (0.7183–0.7273) | 0.1465 | 0.4691 | 0.0521 |
| Logistic regression / uncalibrated | 0.7230 (0.7183–0.7273) | 0.2062 | 0.6037 | 0.2340 |
| Random forest / calibrated (sigmoid) | 0.7797 (0.7752–0.7845) | 0.1358 | 0.4316 | 0.0158 |
| Random forest / uncalibrated | 0.7797 (0.7752–0.7845) | 0.1570 | 0.4870 | 0.1372 |

ROC-AUC measures ranking; higher is better. Brier, log loss and ECE measure probability error; lower is better.
These ranges describe sensitivity to random splits. They are not confidence intervals: the test sets overlap across seeds.
Classification precision, recall and F1 use the configured 0.5 threshold, not an optimised lending threshold.
This is historical Taiwanese credit-card data, not live borrower data or evidence of current lending performance.

[Explanations for real held-out rows, seed 42](seed-42/explanations/README.md)
Tree SHAP explains the uncalibrated gradient-boosting score in log-odds. Calibrated probabilities are shown separately.
