import os
import sys
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import process_data
from src.evaluation import evaluate_model

def train_pipeline():
    # 1. Run preprocessing to ensure data is ready for training
    print("Running data preprocessing pipeline...")
    process_data()
    
    # 2. Load the active configuration parameters
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 3. Configure our local MLflow database tracking connection
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Classifier")

    # 4. Read processed datasets
    train_data = pd.read_csv("data/train_processed.csv") if os.path.exists("data/train_processed.csv") else pd.read_csv("data/heart_disease.csv")
    test_data = pd.read_csv("data/test_processed.csv") if os.path.exists("data/test_processed.csv") else pd.read_csv("data/heart_disease.csv")

    target_col = config["data"].get("target_col", "target")
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # 5. Start the explicit MLflow tracking context
    with mlflow.start_run():
        # Read the unique DVC asset hash value
        dvc_hash = "unknown"
        if os.path.exists("data/heart_disease.csv.dvc"):
            with open("data/heart_disease.csv.dvc", "r") as f:
                for line in f:
                    if "md5:" in line:
                        dvc_hash = line.split("md5:")[-1].strip()

        # Log hyperparameters
        mlflow.log_param("type", "RandomForest")
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])
        mlflow.log_param("dvc_data_version", dvc_hash)

        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # 6. Call the new evaluation module
        metrics = evaluate_model(model, X_test, y_test)
        
        # 7. Log model
        mlflow.sklearn.log_model(model, "model")
        
        # 8. Threshold Enforcement
        acc = metrics["accuracy"]
        print(f"Run Logged Successfully: Accuracy={acc:.4f} | Trees={config['model']['n_estimators']}")
        
        min_acc = config["model"].get("min_accuracy_threshold", 0.70)
        if acc < min_acc:
            print(f"Threshold check failed! Accuracy {acc:.4f} < {min_acc}")
            sys.exit(1)
        else:
            print(f"Performance threshold met (Accuracy: {acc:.4f})")

if __name__ == "__main__":
    train_pipeline()