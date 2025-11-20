def elasticnet_regression_model(x_train, y_train, x_test, y_test):
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.1, 0.5, 0.9]
    }
    grid = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    metrics = {
        "Model": "ElasticNet Regression",
        "Best Params": grid.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }
    return best_model, pd.DataFrame([metrics])
