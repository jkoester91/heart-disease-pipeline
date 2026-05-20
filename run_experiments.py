import os
import sys
# Tell Python to look at the current root directory for the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
from src.preprocessing import process_data  # <-- Add this import
from src.train import train_pipeline

def run_parameter_sweep():
    # 1. Automatically pre-process the data first to generate the clean CSVs
    print("Running data preprocessing pipeline...")
    process_data() 

    # Define 5 distinct parameter combinations to evaluate performance
    sweeps = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},  # Baseline
        {"n_estimators": 150, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 80, "max_depth": 4}
    ]
    
    # Load the base configurations safely
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f" Initializing automated grid sweep across {len(sweeps)} configurations...")

    for i, params in enumerate(sweeps, 1):
        print(f"\n--- Running Experiment Configuration {i}/{len(sweeps)} ---")
        print(
            f"Hyperparameters: Trees={params['n_estimators']}, Max Depth={params['max_depth']}")

        # Dynamically inject parameters into the operational config frame
        config["model"]["n_estimators"] = params["n_estimators"]
        config["model"]["max_depth"] = params["max_depth"]

        # Write temporary configurations out so src/train.py reads them naturally
        with open("configs/config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        # Trigger the modular pipeline training workflow execution
        train_pipeline()


if __name__ == "__main__":
    run_parameter_sweep()
