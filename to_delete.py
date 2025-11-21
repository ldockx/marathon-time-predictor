import joblib
import pandas as pd
import numpy as np

from src.features.features_creation import (
    config,
    drop_irrelevant_columns,
    one_hot_encode,
    extract_numeric_column,
    ensure_numeric_columns_are_numeric,
    fill_na_with_median,
)


# helper functions
def load_artifacts(model_path, scaler_path="models/scaler.pkl"):
 
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load artifacts: {e}")


def preprocess_new_data(df_new, scaler):

    df = df_new.copy()
    
    df = drop_irrelevant_columns(df, config.get("columns_to_drop", []))
    df = one_hot_encode(df, config.get("columns_to_encode", []))
    df = extract_numeric_column(df, config.get("columns_to_numeric", []))
    df = ensure_numeric_columns_are_numeric(
        df, config.get("columns_to_ensure_numeric", [])
    )
    df = fill_na_with_median(df)

    # Ensure all training columns exist
    training_columns = joblib.load("models/training_columns.pkl")
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    # Scale numeric columns
    scale_cols = config.get("columns_to_scale", [])
    if scale_cols:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df


def predict_new(df_new, model_path="models/xgboost.pkl"):

    model, scaler = load_artifacts(model_path)
    processed = preprocess_new_data(df_new, scaler)
    preds = model.predict(processed)
    return preds


def format_hours(hours_float):

    h = int(hours_float)
    total_minutes = (hours_float - h) * 60
    m = int(total_minutes)
    s = int(round((total_minutes - m) * 60))
    return f"{h}h {m}m {s}s"


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    new_input = pd.DataFrame([{
        "Category": "MAM",
        "km4week": 50,
        "sp4week": 10,
        "CrossTraining": "",
        "Wall21": 2.08,
        "Name": "Lode Dockx",
        "id": 999,
        "Marathon": "Bruges",
        "CATEGORY": "D"
    }])

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
