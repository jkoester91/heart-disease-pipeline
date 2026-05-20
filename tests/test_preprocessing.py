import pytest
import pandas as pd
import numpy as np
from src.preprocessing import load_data, handle_missing_values, encode_categories, process_and_split


@pytest.fixture
def mock_raw_data():
    """Generates a consistent dataframe containing mixed data types and missing blocks."""
    return pd.DataFrame({
        "age": [52, 43, np.nan, 61],
        "sex": ["male", "female", "male", "male"],
        "target": [1, 0, 1, 0]
    })


def test_load_data_invalid_path():
    """Confirms loading handles empty path inputs with an explicit value error exception."""
    with pytest.raises(ValueError):
        load_data("")


def test_handle_missing_values_median(mock_raw_data):
    """Verifies missing numerical measurements drop down to historical column medians."""
    cleaned_df = handle_missing_values(mock_raw_data, numeric_cols=[
                                       "age"], strategy="median")
    assert cleaned_df["age"].isna().sum() == 0
    # Median calculation of 43, 52, 61
    assert cleaned_df["age"].iloc[2] == 52.0


def test_encode_categories(mock_raw_data):
    """Confirms raw categorical strings translate into clean numerical label arrays."""
    encoded_df = encode_categories(mock_raw_data, categorical_cols=["sex"])
    assert encoded_df["sex"].dtype != object
    # 'male' maps cleanly to index integer
    assert encoded_df["sex"].iloc[0] == 1


def test_process_and_split(mock_raw_data):
    """Ensures test/train split boundaries obey the specified split ratios."""
    X_train, X_test, y_train, y_test = process_and_split(
        mock_raw_data, target_col="target", test_size=0.25, random_state=42
    )
    assert len(X_train) == 3
    assert len(X_test) == 1
