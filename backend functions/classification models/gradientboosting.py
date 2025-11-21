import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *

def tune_gradient_boosting(x_train, y_train, x_test, y_test):

    # Simple Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    metrics = {
        "Model": "Gradient Boosting Classifier",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

    return model, pd.DataFrame([metrics])
