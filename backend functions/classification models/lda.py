import sys
sys.path.append("backend functions/functionalities")
from importlibraries import *

def tune_lda(x_train, y_train, x_test, y_test):

    # Simple Linear Discriminant Analysis model (no tuning)
    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    metrics = {
        "Model": "Linear Discriminant Analysis",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

    return model, pd.DataFrame([metrics])
