import joblib
import pandas as pd
import numpy as np
from src.utils.helper_functions import (
    predict_new,
    format_hours
)

if __name__ == "__main__":

    # my data
    # this can of course be improved with input(), or with api or whatever
    new_input = pd.DataFrame([{
        "Category": "MAM",
        "km4week": 23.94,
        "sp4week": 10.71,
        "CrossTraining": "",
        "Wall21": 2.08,
        "Name": "Lode Dockx",
        "id": 999,
        "Marathon": "Bruges",
        "CATEGORY": "D"
    }])

    #can be extended easily to other models
    models = [
        ("XGBoost", "models/xgboost.pkl"),
        ("Random Forest", "models/random_forest.pkl"),
        ("Linear Regression", "models/linear_regression.pkl"),
    ]

    for model_name, model_path in models:
        try:
            pred = predict_new(new_input, model_path=model_path)
            print(f"According to the {model_name} predictor, you ran or will run the marathon in: {format_hours(pred[0])}")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")

    print("My actual marathon time was 4:14:41. I started very well but crashed the second part. All be it, the linear regressor did perform okay.")
