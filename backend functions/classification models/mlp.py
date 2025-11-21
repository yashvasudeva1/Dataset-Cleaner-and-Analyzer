import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *

def tune_mlp(x_train, y_train, x_test, y_test):

    # Simple MLP Classifier
    model = MLPClassifier(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    metrics = {
        "Model": "MLP Classifier",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

    return model, pd.DataFrame([metrics])
