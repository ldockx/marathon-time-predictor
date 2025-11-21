from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

"""
This file sets up the XGBoost regressor model
"""

def create_xgb_model():
    return XGBRegressor()


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def xgb_pipeline(X_train, y_train, X_test, y_test):

    model = create_xgb_model()

    model = train_model(model, X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        **metrics,
    }