from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

""""
This file sets up the linear regressor model
"""

def create_linear_regression_model():
    return LinearRegression()


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


def linear_regression_pipeline(X_train, y_train, X_test, y_test):

    model = create_linear_regression_model()

    model = train_model(model, X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        **metrics,
    }