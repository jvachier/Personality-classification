"""Pytest configuration and fixtures for the test suite."""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add dash_app to path for Dash testing
sys.path.insert(0, str(Path(__file__).parent.parent / "dash_app"))

# Import Dash app components for testing
try:
    from dash_app.src import PersonalityClassifierApp

    DASH_AVAILABLE = True
except ImportError:
    PersonalityClassifierApp = None
    DASH_AVAILABLE = False


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample personality classification data."""
    np.random.seed(42)

    # Generate synthetic personality data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=6,  # Six personality types
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Create feature names that mimic personality traits
    feature_names = [
        "openness_1",
        "openness_2",
        "openness_3",
        "conscientiousness_1",
        "conscientiousness_2",
        "conscientiousness_3",
        "extraversion_1",
        "extraversion_2",
        "extraversion_3",
        "agreeableness_1",
        "agreeableness_2",
        "agreeableness_3",
        "neuroticism_1",
        "neuroticism_2",
        "neuroticism_3",
        "mixed_1",
        "mixed_2",
        "mixed_3",
        "mixed_4",
        "mixed_5",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["personality_type"] = y

    return df


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature data without target."""
    np.random.seed(42)

    X, _ = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,  # Increase informative features for 6 classes
        n_classes=6,
        random_state=42,
    )

    feature_names = [
        "openness_1",
        "openness_2",
        "openness_3",
        "conscientiousness_1",
        "conscientiousness_2",
        "conscientiousness_3",
        "extraversion_1",
        "extraversion_2",
        "extraversion_3",
        "agreeableness_1",
        "agreeableness_2",
        "agreeableness_3",
        "neuroticism_1",
        "neuroticism_2",
        "neuroticism_3",
        "mixed_1",
        "mixed_2",
        "mixed_3",
        "mixed_4",
        "mixed_5",
    ]

    return pd.DataFrame(X, columns=feature_names)


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)

    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=6,
        n_informative=15,  # Increased from default 2 to support 6 classes
        n_redundant=3,
        n_clusters_per_class=1,  # Reduced from default 2 to fit constraint
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def config_dict() -> dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        "data": {
            "train_path": "data/train.csv",
            "test_path": "data/test.csv",
            "target_column": "personality_type",
            "feature_columns": ["openness_1", "conscientiousness_1", "extraversion_1"],
        },
        "model": {
            "type": "random_forest",
            "params": {"n_estimators": 100, "random_state": 42, "max_depth": 10},
        },
        "training": {"validation_split": 0.2, "cv_folds": 5, "random_state": 42},
    }


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        "DATA_PATH": "/tmp/test_data",
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Custom assertions for ML testing
def assert_model_performance(y_true, y_pred, min_accuracy: float = 0.5):
    """Assert that model performance meets minimum requirements."""
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= min_accuracy, (
        f"Model accuracy {accuracy:.3f} below minimum {min_accuracy}"
    )


def assert_data_shape(
    df: pd.DataFrame, expected_rows: int | None = None, expected_cols: int | None = None
):
    """Assert that DataFrame has expected shape."""
    if expected_rows is not None:
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    if expected_cols is not None:
        assert len(df.columns) == expected_cols, (
            f"Expected {expected_cols} columns, got {len(df.columns)}"
        )


def assert_no_missing_values(df: pd.DataFrame):
    """Assert that DataFrame has no missing values."""
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in DataFrame"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "mlops: marks tests as MLOps related")


# Dash application fixtures (conditionally loaded)
@pytest.fixture
def dash_app():
    """Create a Dash application for testing."""
    if not DASH_AVAILABLE:
        pytest.skip("Dash application not available")

    test_app = PersonalityClassifierApp(
        model_name="test_model", model_stage="Development"
    )
    return test_app


@pytest.fixture
def dash_client(dash_app):
    """Create a test client for the Dash application."""
    if not DASH_AVAILABLE:
        pytest.skip("Dash application not available")

    return dash_app.get_app().server.test_client()
