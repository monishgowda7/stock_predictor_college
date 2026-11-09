# Model Evaluation 📊

This document summarizes the performance of all machine learning models evaluated during the project.

Multiple feature versions were tested to understand how feature engineering and feature selection affected model performance.

The experiments were divided into:

- V1: Original market features (baseline)
- V2: All engineered technical indicators
- V3: Selected engineered features (Top 25 features)

---

# Evaluation Metrics

The following metrics were used:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct upward movement predictions |
| Recall | Ability to detect upward movements |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Classification separation ability |

---

# Version 1: Baseline Models 📌

## Original Features

The first experiment used only the original market features without additional technical indicators.

Purpose:

- Establish a baseline performance
- Understand model behaviour with raw market information

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-|-|-|-|-|-|
| Logistic Regression | 59.79% | 63.64% | 65.79% | 64.70% | 62.08% |
| Decision Tree | 55.37% | 63.92% | 46.62% | 53.91% | 56.56% |
| Random Forest | 53.47% | 62.57% | 42.11% | 50.34% | 57.62% |
| XGBoost | 54.95% | 60.40% | 56.77% | 58.53% | 58.92% |
| LightGBM | 56.63% | 60.42% | 65.41% | 62.82% | 57.38% |

### Observation

Logistic Regression achieved the strongest baseline performance with the highest ROC-AUC and F1 score.

Tree-based models showed lower performance, indicating that raw market features alone may not contain enough nonlinear patterns.

---

# Tuned Logistic Regression ⚙️

Logistic Regression was optimized using hyperparameter tuning.

Selected parameters:
C = 0.01
solver = lbfgs
max_iter = 1000

Performance:

| Model | Accuracy | Precision | Recall | F1 |
|-|-|-|-|-|
| Tuned Logistic Regression | 59.37% | 61.23% | 74.81% | 67.34% |

### Observation

Tuning improved recall significantly.

The model detected more upward movements, but accuracy slightly decreased.

---

# Version 2: All Engineered Features 🔧

This experiment introduced technical indicators:

- SMA
- EMA
- RSI
- MACD
- Bollinger Bands

Purpose:

- Test whether additional market signals improve prediction

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-|-|-|-|-|-|
| Logistic Regression | 53.83% | 52.48% | 80.87% | 63.66% | 55.49% |
| XGBoost | 51.91% | 58.97% | 12.57% | 20.72% | 52.33% |
| CatBoost | 54.92% | 60.98% | 27.32% | 37.74% | 56.27% |

### Observation

Adding all engineered features did not improve performance.

Possible reasons:

- Increased feature noise
- Redundant indicators
- Market randomness

This motivated feature selection.

---

# Version 3: Selected Engineered Features 🔍

Top 25 features were selected using Random Forest feature importance.

Purpose:

- Remove unnecessary features
- Reduce noise
- Improve generalization

---

## Initial Models

| Model | Accuracy | F1 | ROC-AUC |
|-|-|-|-|
| CatBoost | 57.65% | 50.16% | 58.73% |
| XGBoost | 55.19% | 46.05% | 58.43% |

---

## Tuned Models ⚙️

### XGBoost Tuned

Parameters:
n_estimators = 1000
learning_rate = 0.01
max_depth = 3
min_child_weight = 3
gamma = 0.1
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 2

Results:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|-|-|-|-|-|
|58.47%|59.63%|52.46%|55.81%|59.37%|

---

### CatBoost Tuned

Results:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|-|-|-|-|-|
|56.83%|59.26%|43.72%|50.31%|59.28%|

---

### LightGBM Tuned

Results:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|-|-|-|-|-|
|54.92%|57.63%|37.16%|45.18%|56.44%|

---

# Final Model Comparison 🏆

| Model | Best Version | Accuracy | F1 Score | ROC-AUC |
|-|-|-|-|-|
| Logistic Regression Tuned | V1 | 59.37% | 67.34% | - |
| Logistic Regression Baseline | V1 | 59.79% | 64.70% | 62.08% |
| XGBoost Tuned | V3 | 58.47% | 55.81% | 59.37% |
| CatBoost Initial | V3 | 57.65% | 50.16% | 58.73% |
| LightGBM Baseline | V1 | 56.63% | 62.82% | 57.38% |

---

# Key Findings 💡

1. Feature engineering did not automatically improve performance.

2. More features introduced additional noise.

3. Feature selection helped tree-based models recover performance.

4. Logistic Regression performed strongly because market direction behaved closer to a linear classification problem.

5. Predicting financial markets remains challenging due to randomness and external factors.

---

# Future Improvements 🚀

Future experiments could include:

- Financial news sentiment analysis
- Macroeconomic indicators
- Ensemble models
- Deep learning approaches
- Real-time prediction systems
