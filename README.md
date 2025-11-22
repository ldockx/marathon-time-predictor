# 🏃‍♂️ Marathon Time Predictor

Ever wondered how long it would take you to run a marathon? I did, so I built a marathon time prediction tool using machine learning.

This repo is all about taking a dataset of marathon runners and turning it into a small, interactive ML application.

## 🎯 What this repo does

Fetches marathon runner data from Kaggle using the Kaggle API

Cleans the data and performs basic feature engineering

Builds 3 regression models: Linear Regression, Random Forest Regression, and XGBoost

Evaluates and compares models using multiple regression metrics

Saves models for reuse and allows interactive predictions from user input

Note: The dataset is limited and this is a very naive approach, but the goal is to showcase building a complete ML workflow. Fine-tuning and advanced modeling can be done in future iterations.

## 🛠️ How to use

Train models:
Run main.py → models are trained, evaluated, compared, and saved.

Predict your marathon time:
Run predict.py → a small interactive window will pop up asking for your input (age, pace, etc.) and the tool will return predictions from all 3 models.
