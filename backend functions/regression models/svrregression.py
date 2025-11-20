def svr_regression_model(x_train, y_train, x_test, y_test):
    param_grid = {
        "kernel": ["linear", "poly", "rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"]
    }
    grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    metrics = {
        "Model": "SVR",
        "Best Params": grid.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }
    return best_model, pd.DataFrame([metrics])
