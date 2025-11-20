from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

"""
This file sets up the XGBoost regressor model
"""

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, **kwargs):

    # Initialize model
    model = XGBRegressor(**kwargs)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
