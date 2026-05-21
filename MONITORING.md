# Drift Analysis

### 1. Which features showed drift?
Based on the Evidently report, the following features showed statistically significant drift (K-S test):
- **thalach (Maximum Heart Rate Achieved)**
- **age (Patient Age)**

### 2. Why did they drift?
The statistical drift indicates that the current patient population distribution deviates from the training reference data. This suggests that the current environment is dealing with a different demographic or cardiac profile than the one the model was originally trained on.

### 3. Impact on model performance
Because these features are core inputs to the model, this drift is likely to reduce the model's reliability. We expect a decrease in prediction accuracy (Precision and Recall), as the model's decision boundaries are no longer optimized for the current patient data distribution.

### 4. Recommended Action
To mitigate this impact, I recommend the following:
- **Model Retraining:** Trigger an automated retraining pipeline using the most recent production data to ensure the model aligns with current patient demographics.
- **Data Quality Audit:** Investigate if there have been changes in the upstream data collection process for `thalach` or `age` to rule out data quality issues.