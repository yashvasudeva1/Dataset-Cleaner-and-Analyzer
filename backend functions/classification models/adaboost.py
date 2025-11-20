import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *def tune_adaboost(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier()
    params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    metrics = {
        "Model": "AdaBoost Classifier",
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }
    return best_model, pd.DataFrame([metrics])


