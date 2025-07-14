"""Tests for model building functionality."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from modules.model_builders import build_stack, build_stack_c, build_sklearn_stack


class TestModelBuilders:
    """Test cases for model builder functions."""

    @pytest.fixture
    def mock_trial(self):
        """Create a mock optuna trial object."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.1
        trial.suggest_int.return_value = 100
        trial.suggest_categorical.return_value = 'gini'
        return trial

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
        return X

    def test_build_stack_returns_pipeline(self, mock_trial, sample_data):
        """Test that build_stack returns a sklearn Pipeline."""
        pipeline = build_stack(mock_trial, seed=42, wide_hp=False)

        # Should return a pipeline-like object
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')

    def test_build_stack_c_returns_pipeline(self, mock_trial, sample_data):
        """Test that build_stack_c returns a sklearn Pipeline."""
        pipeline = build_stack_c(mock_trial, seed=42)

        # Should return a pipeline-like object
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')

    def test_build_sklearn_stack_returns_pipeline(self, mock_trial, sample_data):
        """Test that build_sklearn_stack returns a sklearn Pipeline."""
        pipeline = build_sklearn_stack(mock_trial, seed=42, X_full=sample_data)

        # Should return a pipeline-like object
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')

    def test_build_stack_with_wide_hp(self, mock_trial, sample_data):
        """Test build_stack with wide hyperparameter search."""
        pipeline = build_stack(mock_trial, seed=42, wide_hp=True)

        # Should still return a valid pipeline
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')

    def test_build_functions_call_trial_methods(self, mock_trial, sample_data):
        """Test that build functions properly call trial suggestion methods."""
        build_stack(mock_trial, seed=42, wide_hp=False)

        # Trial methods should have been called
        assert mock_trial.suggest_float.called or mock_trial.suggest_int.called or mock_trial.suggest_categorical.called
