import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *

def random_forest_regression_model(x_train, y_train, x_test, y_test):

    # Simple Random Forest Regression Model (default parameters)
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    metrics = {
        "Model": "Random Forest Regressor",
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }

    return model, pd.DataFrame([metrics])
