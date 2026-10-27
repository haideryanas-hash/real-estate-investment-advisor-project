# Evaluation report (held-out test set, n=7,072)

## Classification - XGBoost
| Metric | Value |
|---|---|
| Accuracy | 0.9924 |
| Precision (good) | 0.9891 |
| Recall (good) | 0.9873 |
| F1 (good) | 0.9882 |
| ROC-AUC | 0.9996 |

## Regression - XGBoost
| Metric | Model | Fixed-8% baseline |
|---|---|---|
| RMSE (lakhs) | 1.735 | 23.329 |
| MAE (lakhs) | 1.342 | 16.429 |
| R2 | 0.99994 | 0.98828 |
| MAPE % | 0.551 | 4.197 |

RMSE improvement over the baseline: 92.6%.
90% prediction interval (relative error): -1.13% to +1.11%.

## Honesty checks
* **Signal test** - predicting TODAY's price from all other features: R2 = -0.1462, RMSE = 151.6 L (price std = 141.6 L). The features contain no usable information about price.
* **Ablation** - classifier without the four price-based features: ROC-AUC 0.9568 (full model 0.9996), accuracy 0.8855 (full 0.9924).
