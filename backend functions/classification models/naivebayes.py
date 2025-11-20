
def tune_naive_bayes(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    metrics = {
        "Model": "Gaussian Naive Bayes",
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }
    return best_model, pd.DataFrame([metrics])
