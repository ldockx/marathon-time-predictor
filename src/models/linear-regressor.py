from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

""""
This file sets up the linear regressor model
"""


def train_and_evaluate_linear_regression(X_train, y_train, X_test, y_test):

    # Initialize model
    model = LinearRegression()

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
