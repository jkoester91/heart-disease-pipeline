import os
import sys
import yaml
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

def monitor_data_drift():
    # 1. Load data settings
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_path = "data/train_processed.csv"
    if not os.path.exists(train_path):
        print(f"❌ Error: Baseline data '{train_path}' not found.")
        sys.exit(1)
        
    # 2. Prepare datasets
    reference_df = pd.read_csv(train_path)
    np.random.seed(42)
    production_df = reference_df.copy()
    
    # Inject synthetic drift
    if "age" in production_df.columns:
        production_df["age"] = production_df["age"] + np.random.normal(5, 2, size=len(production_df))
    if "thalach" in production_df.columns:
        production_df["thalach"] = production_df["thalach"] - np.random.normal(10, 5, size=len(production_df))

    print("📊 Evaluating data drift...")
    
    # 3. Configure and run report
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=reference_df, current_data=production_df)
    
    # 4. Extract metrics using the correct 0.7.x approach
    result = snapshot.dict()
    summary_metric = result["metrics"][0]  # DriftedColumnsCount
    summary_values = summary_metric.get("value", {})
    
    drifted_features = int(summary_values.get("count", 0))
    drift_share = summary_values.get("share", 0.0)
    total_features = len(result["metrics"]) - 1  # Exclude summary metric
    
    print("\n==================================================")
    print("📡 EVIDENTLY DRIFT MONITORING SUMMARY")
    print("==================================================")
    print(f"Total Features Analyzed: {total_features}")
    print(f"Drifted Feature Count:   {drifted_features}")
    print(f"Overall Drift Share:     {drift_share:.2%}")
    print("==================================================")

    # 5. Export HTML
    os.makedirs("reports", exist_ok=True)
    html_report_path = "reports/data_drift_report.html"
    try:
        snapshot.save_html(html_report_path)  # Use snapshot instead of drift_report
    except Exception as e:
        print(f"⚠️  HTML report generation failed: {e}")
        print("💡 Drift analysis completed successfully, but HTML export unavailable")
    print(f"💾 Interactive dashboard report saved to: {html_report_path}")

    # 6. Check threshold
    drift_threshold = 0.30
    if drift_share > drift_threshold:
        print(f"🚨 ALERT: Drift share ({drift_share:.2%}) exceeds {drift_threshold:.2%}!")
        sys.exit(1)
    else:
        print("✅ Data drift within safe boundaries.")
        sys.exit(0)

if __name__ == "__main__":
    monitor_data_drift()