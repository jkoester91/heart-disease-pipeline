import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    # 1. Generate predictions
    predictions = model.predict(X_test)
    
    # 2. Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="binary", zero_division=0),
        "recall": recall_score(y_test, predictions, average="binary", zero_division=0),
        "f1_score": f1_score(y_test, predictions, average="binary", zero_division=0)
    }
    
    # 3. Log to MLflow
    mlflow.log_metrics(metrics)
    
    # 4. Return metrics to train.py for threshold enforcement
    return metrics