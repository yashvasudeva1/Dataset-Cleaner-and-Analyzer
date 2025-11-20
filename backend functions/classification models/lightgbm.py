import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *
def tune_lightgbm(x_train, y_train, x_test, y_test):
    model = LGBMClassifier()
    params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': [31, 63, 127]}
    grid = RandomizedSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, n_iter=5, random_state=42)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    metrics = {
        "Model": "LightGBM Classifier",
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }
    return best_model, pd.DataFrame([metrics])

