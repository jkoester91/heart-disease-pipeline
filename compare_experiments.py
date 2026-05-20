import mlflow


def evaluate_best_run():
    # Connect to the local MLflow backend tracking database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Query all logged records from the specific experiment run list
    experiment = mlflow.get_experiment_by_name("Heart_Disease_Classifier")
    if not experiment:
        print("No recorded experiments found. Please populate runs first.")
        return

    # Use programmatic search to pull logs sorted by highest accuracy performance
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )

    if runs_df.empty:
        print("No completed runs found inside the database logs.")
        return

    # Extract the top-performing model slice
    best_run = runs_df.iloc[0]

    print("\n==================================================")
    print("BEST PERFORMING EXPERIMENT RUN RUN DETECTED")
    print("==================================================")
    print(f"Run ID:      {best_run['run_id']}")
    print(f"Accuracy:    {best_run['metrics.accuracy']:.4f}")
    print(f"F1-Score:    {best_run['metrics.f1_score']:.4f}")
    print(
        f"Model Type:  {best_run['params.type']}" if 'params.type' in best_run else "Model Type: N/A")
    print(
        f"Trees (n):   {best_run['params.n_estimators']}" if 'params.n_estimators' in best_run else "Trees: N/A")
    print(
        f"Max Depth:   {best_run['params.max_depth']}" if 'params.max_depth' in best_run else "Depth: N/A")
    print("==================================================\n")


if __name__ == "__main__":
    evaluate_best_run()
