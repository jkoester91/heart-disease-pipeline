# Drift Monitoring Analysis

### 1. Which features showed drift and why?
The features `age` and `thalach` (maximum heart rate) showed significant drift. This was simulated to represent a shift in the patient demographic or potential changes in medical measurement equipment calibration in a clinical setting.

### 2. Would this drift likely affect model performance?
Yes. Since the model relies heavily on `thalach` for predicting heart disease risk, a shift in the distribution of this feature compared to the training data will reduce the model's predictive confidence and accuracy.

### 3. Recommended Actions
- **Retrain:** The model should be retrained on the updated "production" data to capture the new distribution.
- **Monitor:** Continue monitoring these features to determine if this is a one-time event or a sustained shift in the patient population.