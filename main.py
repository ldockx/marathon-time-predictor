import os
import pandas as pd
import joblib
from src.features.features_creation import create_features
from src.models.linear_regressor import linear_regression_pipeline
from src.models.random_forest_regressor import random_forest_pipeline
from src.models.xgb_regressor import xgb_pipeline
from src.utils.helper_functions import save_model, compare_model_performance


def main():

    X_train, X_test, y_train, y_test, scaler = create_features()

    #training columns should be saved for making new predictions
    joblib.dump(X_train.columns.tolist(), "models/training_columns.pkl")

    print("Training models\n")

    lr_results = linear_regression_pipeline(X_train, y_train, X_test, y_test)
    rf_results = random_forest_pipeline(X_train, y_train, X_test, y_test)
    xgb_results = xgb_pipeline(X_train, y_train, X_test, y_test)

    print("Saving models\n")

    save_model(lr_results["model"], "linear_regression.pkl")
    save_model(rf_results["model"], "random_forest.pkl")
    save_model(xgb_results["model"], "xgboost.pkl")

    print("Saving scaler for future\n")

    save_model(scaler, "scaler.pkl")

    print("Collecting results\n")

    results = {
        "Linear Regression": {
            "RMSE": lr_results["rmse"],
            "MSE": lr_results["mse"],
            "R2": lr_results["r2"],
        },
        "Random Forest": {
            "RMSE": rf_results["rmse"],
            "MSE": rf_results["mse"],
            "R2": rf_results["r2"],
        },
        "XGBoost": {
            "RMSE": xgb_results["rmse"],
            "MSE": xgb_results["mse"],
            "R2": xgb_results["r2"],
        },
    }

    compare_model_performance(results)

main()