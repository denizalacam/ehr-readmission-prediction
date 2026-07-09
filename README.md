# EHR Readmission Prediction

### End-to-End Machine Learning Pipeline for Predicting 30-Day Hospital Readmission Using Synthetic Electronic Health Record (EHR) Data

EHRPredict is an end-to-end healthcare AI project that predicts **30-day hospital readmission** using synthetic Electronic Health Record (EHR) data generated with **Synthea**.

The project demonstrates the complete machine learning workflow—from transforming raw relational healthcare data into an encounter-level prediction dataset, through feature engineering, model development, evaluation, and comparison of multiple machine learning approaches.

---

# Project Highlights

- Built an encounter-level prediction dataset from relational EHR tables.
- Generated 30-day readmission labels using longitudinal patient encounters.
- Engineered demographic, encounter, utilization, and healthcare cost features.
- Developed a reusable preprocessing pipeline.
- Implemented three predictive models:
  - Logistic Regression
  - PyTorch Multi-Layer Perceptron (MLP)
  - XGBoost
- Compared model performance using multiple evaluation metrics.
- Generated feature importance, ROC curves, confusion matrices, and automated comparison reports.

---

# Workflow

```text
Raw Synthea EHR Data
        │
        ▼
Data Cleaning & Integration
        │
        ▼
30-Day Readmission Label Generation
        │
        ▼
Feature Engineering
        │
        ▼
Train / Test Split
        │
        ▼
Preprocessing Pipeline
(Standardization + One-Hot Encoding)
        │
        ├───────────────┬─────────────────┐
        ▼               ▼                 ▼
Logistic Regression  PyTorch MLP     XGBoost
        │               │                 │
        └───────────────┴─────────────────┘
                        │
                        ▼
               Model Evaluation
                        │
                        ▼
               Model Comparison
```

---

# Models

## Logistic Regression

A classical linear baseline model used to establish benchmark performance.

## PyTorch Neural Network

A Multi-Layer Perceptron (MLP) implemented in PyTorch.

Architecture:

- 19 engineered input features
- Hidden layer (32 neurons)
- ReLU activation
- BCEWithLogitsLoss
- Adam optimizer

## XGBoost

Gradient-boosted decision tree model.

Configuration:

- 300 estimators
- Maximum depth = 4
- Learning rate = 0.05
- Subsample = 0.9
- Column sampling = 0.9

---

# Results

| Metric | Logistic Regression | PyTorch MLP | XGBoost | Best Model |
|--------|--------------------:|------------:|---------:|------------|
| Accuracy | 0.7635 | 0.7815 | **0.8152** | 🏆 XGBoost |
| ROC-AUC | 0.8326 | 0.8537 | **0.8818** | 🏆 XGBoost |
| Precision | 0.7090 | 0.7663 | **0.8047** | 🏆 XGBoost |
| Recall | 0.6584 | 0.6253 | **0.6893** | 🏆 XGBoost |
| F1-Score | 0.6827 | 0.6887 | **0.7425** | 🏆 XGBoost |

Rather than assuming a single algorithm is universally superior, this project evaluates multiple modeling approaches using complementary performance metrics.

Among the evaluated models, **XGBoost achieved the strongest overall predictive performance**, demonstrating the effectiveness of gradient-boosted decision trees for structured healthcare data.

---

# Visualizations

The pipeline automatically generates:

- ROC Curve
- Confusion Matrix
- PyTorch Training Loss
- Logistic Regression Feature Importance
- XGBoost Feature Importance
- Model Comparison Table

---

# Technologies

### Languages & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- XGBoost
- Matplotlib
- Joblib

### Machine Learning Concepts

- Feature Engineering
- Data Preprocessing
- Binary Classification
- Gradient Boosting
- Deep Learning
- Model Evaluation
- Healthcare AI

---

# Repository Structure

```text
ehr-readmission-prediction/

data/
├── raw/
└── processed/

models/
├── logistic_regression.pkl
├── pytorch_mlp.pt
├── xgboost_model.pkl
└── preprocessor.pkl

results/
├── metrics.txt
├── pytorch_metrics.txt
├── xgboost_metrics.txt
├── model_comparison.csv
├── roc_curve.png
├── confusion_matrix.png
├── training_loss.png
├── feature_importance.csv
└── xgboost_feature_importance.png

src/
├── preprocess.py
├── features.py
├── data_pipeline.py
├── train_baseline.py
├── train_pytorch.py
├── train_xgboost.py
└── compare_models.py
```

---

## Repository Structure

``` text
DeepEHR/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── logistic_regression.pkl
│   ├── preprocessor.pkl
│   └── pytorch_mlp.pt
├── results/
├── src/
│   ├── data_pipeline.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_baseline.py
│   ├── train_pytorch.py
│   └── compare_models.py
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

## Technologies

Python • Pandas • NumPy • Scikit-learn • PyTorch • Matplotlib • Joblib •
Machine Learning • Deep Learning • Feature Engineering • Healthcare AI

------------------------------------------------------------------------

## Dataset
=======
# Data Source

This project uses **synthetic Electronic Health Record (EHR) data** generated by **Synthea**.

The processed encounter-level dataset is included to enable immediate reproducibility.

Users wishing to recreate the preprocessing pipeline can generate the original synthetic EHR tables using the official Synthea project.

---

# Installation

```bash
git clone https://github.com/denizalacam/ehr-readmission-prediction.git

cd ehr-readmission-prediction

<<<<<<< HEAD
## Installation

Clone the repository:

```bash
git clone https://github.com/denizalacam/deepehr-readmission.git
cd deepehr-readmission
```

Create a virtual environment:

```bash
=======
>>>>>>> 9477661 (Add XGBoost model and feature importance)
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

# Running the Project

```bash
python src/preprocess.py

python src/features.py

python src/train_baseline.py

python src/train_pytorch.py

python src/train_xgboost.py

python src/compare_models.py
```

---

# Project Evolution

## ✅ Version 1.0

- Built encounter-level dataset
- Developed reusable preprocessing pipeline
- Implemented Logistic Regression baseline
- Built a PyTorch Multi-Layer Perceptron
- Compared classical machine learning and deep learning models

## ✅ Version 1.1

- Added XGBoost classifier
- Automated three-model comparison
- Implemented XGBoost feature importance analysis
- Achieved the best predictive performance with gradient boosting

---

# Future Directions

The project will continue to evolve with additional machine learning capabilities.

### Planned Version 1.2

- SHAP explainability
- Global and local feature interpretation
- Enhanced model transparency for healthcare AI

### Planned Version 1.3

- Streamlit web application
- Interactive patient risk prediction
- Real-time model inference

### Planned Version 1.4

- Hyperparameter optimization using Optuna
- Automated model tuning

### Planned Version 1.5

- LightGBM benchmarking
- CatBoost benchmarking
- Comprehensive gradient boosting comparison

### Planned Version 2.0

- Temporal patient modeling
- Longitudinal EHR representation
- LSTM and Transformer architectures

---

# About This Project

This project demonstrates an end-to-end machine learning workflow for healthcare prediction, emphasizing data engineering, feature engineering, reproducible preprocessing, model evaluation, software organization, and comparison of classical machine learning with deep learning approaches.
