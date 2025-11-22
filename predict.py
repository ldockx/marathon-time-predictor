import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from src.utils.helper_functions import (
    predict_new,
    format_hours,
    get_category
)

from src.utils.ui import (
    get_user_input,
    hms_to_hours,
    show_results
)


def marathon_predictor_loop():

    while True:
        # --- GUI Input ---
        values = get_user_input()

        if values is None:
            print("User exited.")
            break

        # Compute derived values
        km4week = values["total_km"] / 4
        sp4week = values["sp4week"]
        wall21 = hms_to_hours(values["hm_time"])
        category = get_category(values["gender"], values["age"])

        new_input = pd.DataFrame([{
            "Category": category,
            "km4week": km4week,
            "sp4week": sp4week,
            "CrossTraining": "",
            "Wall21": wall21,
            "Name": "Runner",
            "id": 999,
            "Marathon": "Unknown",
            "CATEGORY": "D"
        }])

        models = [
            ("XGBoost", "models/xgboost.pkl"),
            ("Random Forest", "models/random_forest.pkl"),
            ("Linear Regression", "models/linear_regression.pkl"),
        ]

        predictions = {}

        for model_name, model_path in models:
            try:
                pred = predict_new(new_input, model_path=model_path)
                predictions[model_name] = format_hours(pred[0])
            except Exception as e:
                predictions[model_name] = f"Error: {e}"

        # Show results in new decorated window
        show_results(predictions)


if __name__ == "__main__":
    marathon_predictor_loop()
