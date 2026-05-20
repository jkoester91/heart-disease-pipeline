# Drift Monitoring Analysis

### 1. Which features showed drift and why?
The features `thalach` (max heart rate) and `oldpeak` showed drift. This was simulated to represent a shift in the patient demographic or a change in measurement equipment calibration.

### 2. Would this drift likely affect model performance?
Yes. Since the model relies heavily on these features for classification, a shift in their distribution compared to the training data distribution will cause the model to make less accurate predictions.

### 3. Recommended Actions
- **Retrain:** The model should be retrained on the updated data to capture the new distribution.
- **Monitor:** Continue monitoring these features to determine if this is a sustained shift in the population or a transient issue.