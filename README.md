# Machine Learning Assignment 2: Predictive Maintenance Multi-Model Classification

## a. Problem Statement
The goal of this project is to implement a predictive maintenance system for industrial machinery using sensor readings. Early detection of component wear and breakdown prevents catastrophic equipment failure, optimizes maintenance schedules, and reduces industrial operational downtime.

## b. Dataset Description
- **Dataset Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset Dimension:** 10,000 instances, 14 attributes
- **Target Variable:** `Machine failure` (Binary: 0 = Healthy / No Failure, 1 = Equipment Breakdown)
- **Feature Selection & Engineering Diligence:** 
  - Raw attributes: `[UDI, Product ID, Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine failure, TWF, HDF, PWF, OSF, RNF]`
  - Identifiers (`UDI`, `Product ID`) were excluded to prevent non-generalizable index memorization.
  - Granular breakdown flags (`TWF, HDF, PWF, OSF, RNF`) represent specific failure sub-modes and were dropped to prevent direct target leakage.
  - Final training & evaluation used the 6 operational sensor and physical attributes: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.
- **Split Strategy:** 80% Training set (8,000 instances) and 20% Stratified Test set (2,000 instances).

## c. Repository and Live Links
- **GitHub Repository Link:** https://github.com/mayureshy25/Machine_Learning_Assignment_2
- **Live Streamlit App Link:** https://machine-learningassignment-2.streamlit.app/

## d. Models Used & Comparative Evaluation

The following models were trained on 8,000 instances and evaluated on an identical 20% stratified test split (2,000 instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9680 | 0.9004 | 0.6667 | 0.1176 | 0.2000 | 0.2712 |
| **Decision Tree** | 0.9730 | 0.9147 | 0.7692 | 0.2941 | 0.4255 | 0.4656 |
| **KNN** | 0.9740 | 0.8292 | 0.8333 | 0.2941 | 0.4348 | 0.4861 |
| **Naive Bayes** | 0.9600 | 0.9053 | 0.2857 | 0.1176 | 0.1667 | 0.1655 |
| **Support Vector Machine** | 0.9720 | 0.9453 | 0.8750 | 0.2059 | 0.3333 | 0.4167 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | High accuracy (96.8%) due to imbalanced classes, but low recall (11.76%), struggling with non-linear boundaries. |
| **Decision Tree** | Captures distinct operational threshold splits (temperature and torque thresholds), yielding balanced F1 (0.4255) and MCC (0.4656). |
| **KNN** | Strong local neighborhood clustering in standardized feature space, achieving highest MCC (0.4861) and F1 (0.4348) with 83.33% precision. |
| **Naive Bayes** | Lowest precision (28.57%) and MCC (0.1655), as conditional independence fails between correlated thermal and mechanical sensors. |
| **Support Vector Machine** | Highest AUC (0.9453) and Precision (87.50%), establishing a clear boundary margin with minimal false alarms. |
| **Overall Winner for your dataset?** | **KNN / Decision Tree** for balanced fault identification (highest MCC and Recall), while **SVM** is the best discriminator for low false alarms (highest ROC-AUC of 0.9453 and Precision of 87.50%). |
