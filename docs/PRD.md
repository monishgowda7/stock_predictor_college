# Project Requirements Document (PRD)

This document defines the objectives, scope, requirements, and engineering decisions for the NIFTY50 Market Direction Prediction project.

---

# 1. Project Overview

Financial markets are influenced by a wide range of domestic and global economic factors, making short-term market movements difficult to predict. This project investigates whether economically motivated domestic and global financial indicators can improve the prediction of the next trading day's NIFTY50 market direction using machine learning.

Rather than focusing solely on achieving high prediction accuracy, this project follows a structured experimental approach by comparing domestic indicators, global indicators, and their combination across multiple machine learning models.

The primary objective is to identify which group of indicators contributes most to predictive performance while building a reproducible, well-documented, and explainable machine learning project.

---

## Project Hypothesis

> Combining domestic and global financial indicators will provide better predictive performance than using either group independently.

# 2. Problem Statement

Predicting the next day's stock market direction is a challenging machine learning problem because financial markets are influenced by numerous domestic and global economic factors. While many indicators are believed to affect market behaviour, it is often unclear which indicators provide the most meaningful predictive information.

This project aims to address this challenge by systematically comparing domestic financial indicators, global financial indicators, and their combination to determine which group contributes most effectively to predicting the next trading day's NIFTY50 market direction.

# 3. Research Question

To what extent does combining domestic and global financial indicators improve the prediction of the next trading day's NIFTY50 market direction compared to using either group independently?

# 4. Project Objectives

- Predict the next trading day's NIFTY50 market direction using machine learning.
- Compare the predictive performance of domestic indicators, global indicators, and their combination.
- Evaluate multiple machine learning models under identical experimental settings.
- Identify the financial indicators that contribute most to prediction performance.
- Build a reproducible, explainable, and well-documented machine learning project.

# 5. Success Criteria

The project will be considered successful if it satisfies the following criteria:

- A complete end-to-end machine learning pipeline is implemented.
- Three experimental groups (Domestic, Global, Combined) are evaluated under the same methodology.
- Multiple machine learning models are trained and compared fairly.
- Every selected financial indicator has a documented justification.
- Results are presented using appropriate evaluation metrics and visualizations.
- The project is fully reproducible with clear documentation and setup instructions.

# 6. Project Scope

## In Scope

- Predict the next trading day's NIFTY50 market direction.
- Use domestic financial indicators.
- Use global financial indicators.
- Compare domestic, global, and combined feature sets.
- Train and evaluate multiple traditional machine learning models.
- Build a reproducible machine learning pipeline.

## Out of Scope

- Individual stock prediction.
- Intraday trading prediction.
- Live trading or automated trading systems.
- Deep learning models.
- Reinforcement learning.
- News sentiment and social media analysis (Version 1).

# 7. Constraints

- The project relies on publicly available historical financial data.
- Only information available before market close will be used for prediction.
- The project focuses exclusively on NIFTY50 market direction.
- Model performance is dependent on the quality and availability of historical data.
- The project is limited to traditional machine learning models in Version 1.

# 8. Risks

- Financial markets are highly dynamic and influenced by unpredictable events.
- Historical relationships between indicators may change over time.
- Some financial indicators may have limited predictive value.
- Data quality or missing values may affect model performance.
- High model accuracy does not guarantee future market performance.

# 9. Deliverables

- A reproducible machine learning pipeline.
- Cleaned and processed financial datasets.
- Comparative analysis of domestic, global, and combined indicators.
- Performance comparison across multiple machine learning models.
- Visualizations explaining model performance and feature importance.
- Complete project documentation.
- Open-source GitHub repository with setup instructions.
