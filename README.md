#Predictive Maintenance Using AI
This repository presents an end-to-end machine learning pipeline to predict failure events in milling machines using synthetic sensor and operational data. The goal is to leverage AI techniques to improve predictive maintenance strategies and minimize unexpected equipment downtimes.

📊 Dataset Overview
The dataset used is a synthetic simulation of milling machine operations and includes 10,000 rows with 14 features:


Feature	Description
UDI	Unique Identifier (1–10,000)
Product ID	Encodes product type (L, M, H) and serial number
Type	Product quality category: L (low), M (medium), H (high)
Air temperature [K]	Generated using a normalized random walk around 300K
Process temperature [K]	Air temp + 10K + noise
Rotational speed [rpm]	Based on 2860W power with noise
Torque [Nm]	Normally distributed around 40Nm
Tool wear [min]	Wear increases by product type (L/M/H: 2/3/5)
Machine failure	Binary indicator of any failure
TWF, HDF, PWF, OSF, RNF	Specific failure modes
📌 Reference:
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020. DOI:10.1109/AI4I49448.2020.00023

📈 Project Pipeline
1. Data Exploration
Descriptive statistics

Distributions of sensor readings

Correlation matrix

PCA for dimensionality visualization

Class imbalance visualization

2. Preprocessing
One-hot encoding for categorical features

Median imputation for numeric columns

Standard scaling of numerical values

3. Handling Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the failure class

4. Initial Model
Classifier: XGBoost

Accuracy: 0.9825

F1 Score: 0.785

Confusion matrix and per-type error analysis

5. Model Tuning with Optuna
Optimized XGBoost hyperparameters

Best params via Bayesian Optimization

Improved accuracy: 0.9695, F1 Score: 0.667

6. Model Comparison
Tested various classifiers including:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

LightGBM

XGBoost

🧪 Tools & Libraries
pandas, numpy for data handling

matplotlib, seaborn for visualization

scikit-learn for ML preprocessing and models

xgboost, lightgbm for boosting models

optuna for hyperparameter tuning

imbalanced-learn for handling class imbalance

