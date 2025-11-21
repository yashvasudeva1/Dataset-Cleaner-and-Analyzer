import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *

def tune_svm(x_train, y_train, x_test, y_test):

    # Simple Support Vector Classifier (default: RBF kernel)
    model = SVC(random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    metrics = {
        "Model": "SVC",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

    return model, pd.DataFrame([metrics])
