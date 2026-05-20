# Heart Disease MLOps Pipeline

This repository implements a production-ready MLOps pipeline for the Heart Disease classification task.

## Features
- **Data Versioning:** Managed with DVC.
- **Experiment Tracking:** Integrated with MLflow to track hyperparameters and metrics.
- **CI/CD:** Automated testing and training via GitHub Actions.
- **Monitoring:** Drift detection using Evidently.

## How to Run
1. **Setup:** `pip install -r requirements.txt`
2. **Run Tests:** `pytest tests/ -o pythonpath=. -v`
3. **Run Pipeline:** `python run_experiments.py`

## CI/CD Pipeline
Every push to the `main` branch triggers the GitHub Actions pipeline, which executes the unit test suite and validates model performance against a minimum accuracy threshold.