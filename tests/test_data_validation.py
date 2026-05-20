import os
import pandas as pd
import pytest

@pytest.fixture
def actual_data():
    """Loads the actual data artifact being used by the training pipeline."""
    data_path = "data/heart_disease.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} not found. Skipping validation tests.")
    return pd.read_csv(data_path)


def test_expected_columns_present(actual_data):
    """1. Verifies all expected structural features are present in the dataset."""
    required_columns = ["age", "sex", "target"]  # Core tracking baseline
    for col in required_columns:
        assert col in actual_data.columns, f"Expected column '{col}' is missing!"


def test_target_is_binary(actual_data):
    """2. Verifies classification target boundary rules contain only binary outcomes."""
    unique_targets = actual_data["target"].dropna().unique()
    for value in unique_targets:
        assert value in [0, 1], f"Unexpected classification outcome detected: {value}"


def test_numeric_feature_ranges(actual_data):
    """3. Verifies numerical attributes fall within realistic physiological bounds."""
    # Ensure age parameters contain normal ranges
    assert actual_data["age"].min() >= 0, "Detected impossible negative age metric."
    assert actual_data["age"].max() <= 120, "Age metric exceeds realistic human baseline limits."