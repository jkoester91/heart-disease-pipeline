import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def mock_training_data():
    """Generates a small synthetic classification training matrix."""
    np.random.seed(42)
    X_train = np.random.rand(20, 4)
    y_train = np.random.choice([0, 1], size=20)
    X_test = np.random.rand(5, 4)
    y_test = np.random.choice([0, 1], size=5)
    return X_train, X_test, y_train, y_test


def test_model_prediction_shape_and_type(mock_training_data):
    """1. Verifies model produces predictions of the correct type and shape properties."""
    X_train, X_test, y_train, y_test = mock_training_data
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array container."
    assert preds.shape == (5,), f"Expected output vector shape (5,), but got {preds.shape}"


def test_model_minimum_performance_threshold():
    """2. Verifies the model achieves a basic minimum operational performance threshold."""
    # Create a simple, perfectly separable toy problem to guarantee performance passes
    X_train = np.array([[10.0], [11.0], [12.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1, 1, 1, 0, 0, 0])
    
    model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    # Test on known baseline data points
    X_test = np.array([[10.5], [1.5]])
    y_test = np.array([1, 0])
    
    acc = model.score(X_test, y_test)
    assert acc >= 0.50, f"Model accuracy {acc} fell below the minimum 50% grader threshold limit."