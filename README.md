# SafeMeds – Drug Interaction Predictor

## Description
SafeMeds predicts whether two drugs will have harmful interactions using Machine Learning and Deep Learning.

## Features
- CatBoost ML model (~86% accuracy)
- LSTM deep learning model
- Explainable AI using SHAP
- Flask-based web application
- User and Admin modules

## Tech Stack
- Python
- Flask
- CatBoost
- TensorFlow/Keras
- SQLite
- HTML, CSS

## How it Works
1. User enters two drugs
2. Features are generated
3. Model predicts interaction risk
4. SHAP explains prediction
5. Result shown on UI

## Setup Instructions
```bash
git lfs install
git clone https://github.com/amyamihh/safemeds.git

## Important Note (Large Files)

This project uses large files such as trained models and datasets.

These files are managed using Git LFS (Large File Storage) and are not stored directly in the repository.

### Before running the project:

Make sure Git LFS is installed:

git lfs install
git lfs pull

### Large files include:
- safemeds_model.cbm (CatBoost model)
- db_drug_interactions.csv (dataset)
- safemeds.db (database)

If files are missing, re-download or retrain using:
train_catboost.py