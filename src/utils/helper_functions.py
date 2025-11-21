import os
import joblib
import pandas as pd
from src.features.features_creation import (
    config,
    drop_irrelevant_columns,
    one_hot_encode,
    extract_numeric_column,
    ensure_numeric_columns_are_numeric,
    fill_na_with_median,
)

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

def compare_model_performance(results_dict):
    """
    Takes a dictionary:
        { "model_name": { "rmse": ..., "mse": ..., "r2": ... }, ... }
    Returns a clean pandas DataFrame.
    """
    comparison = pd.DataFrame(results_dict).T
    print("\n================ Model Performance Comparison ================\n")
    print(comparison)
    print("\n==============================================================\n")
    return comparison

def save_model(model, filename, directory="models"):
    
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(model, path)
    print(f"Model saved: {path}")