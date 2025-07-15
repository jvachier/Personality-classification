"""Tests for dashboard model loader."""

import pytest

from dash_app.dashboard.model_loader import ModelLoader


class TestModelLoader:
    """Test suite for ModelLoader class."""

    def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        loader = ModelLoader(
            model_name="test_model",
            model_version="1.0",
            model_stage="Testing"
        )
        assert loader.model_name == "test_model"
        assert loader.model_version == "1.0"
        assert loader.model_stage == "Testing"
        # Model should be loaded (either real model or dummy)
        assert loader.model is not None

    def test_model_loader_with_ensemble_name(self):
        """Test ModelLoader with ensemble model name."""
        loader = ModelLoader(model_name="ensemble")
        assert loader.model_name == "ensemble"
        assert loader.is_loaded() is True

    def test_model_loader_get_metadata(self):
        """Test model metadata retrieval."""
        loader = ModelLoader(model_name="test_model")
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert "version" in metadata
        assert "stage" in metadata

    def test_model_loader_is_loaded(self):
        """Test model loading status check."""
        loader = ModelLoader(model_name="test_model")
        assert loader.is_loaded() is True

    def test_model_loader_str_representation(self):
        """Test string representation of ModelLoader."""
        loader = ModelLoader(model_name="test_model")
        # Just check that it doesn't raise an error
        str_repr = repr(loader)
        assert isinstance(str_repr, str)


class TestModelPrediction:
    """Test suite for model prediction functionality."""

    @pytest.fixture
    def model_loader(self):
        """Create a ModelLoader for testing predictions."""
        return ModelLoader(model_name="test_model")

    def test_model_prediction_success(self, model_loader):
        """Test successful model prediction."""
        input_data = {
            "Time_spent_Alone": 3.0,
            "Social_event_attendance": 2.0,
            "Going_outside": 4.0,
            "Friends_circle_size": 3.0,
            "Post_frequency": 2.0,
            "Stage_fear_No": 1,
            "Stage_fear_Unknown": 0,
            "Stage_fear_Yes": 0,
            "Drained_after_socializing_No": 1,
            "Drained_after_socializing_Unknown": 0,
            "Drained_after_socializing_Yes": 0,
            "match_p_Extrovert": 0,
            "match_p_Introvert": 0,
            "match_p_Unknown": 1,
        }

        result = model_loader.predict(input_data)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert result["model_name"] == "test_model"

    def test_model_prediction_with_missing_features(self, model_loader):
        """Test prediction with missing input features."""
        input_data = {
            "Time_spent_Alone": 3.0,
            "Social_event_attendance": 2.0
            # Missing other features - should be handled by default values
        }

        result = model_loader.predict(input_data)
        assert isinstance(result, dict)
        assert "prediction" in result

    def test_model_prediction_with_invalid_input(self, model_loader):
        """Test prediction with invalid input data."""
        invalid_input = "invalid_input"

        with pytest.raises((ValueError, TypeError, AttributeError)):
            model_loader.predict(invalid_input)

    def test_model_prediction_empty_input(self, model_loader):
        """Test prediction with empty input."""
        empty_input = {}

        # Should handle empty input with default values
        result = model_loader.predict(empty_input)
        assert isinstance(result, dict)
        assert "prediction" in result


class TestModelLoaderEdgeCases:
    """Test edge cases for ModelLoader."""

    def test_model_loader_with_dummy_fallback(self):
        """Test ModelLoader creates dummy model when no real model found."""
        # Use a model name that won't exist
        loader = ModelLoader(model_name="nonexistent_model")

        # Should still be loaded (with dummy model)
        assert loader.is_loaded() is True
        assert loader.model is not None

        # Metadata should indicate dummy model
        metadata = loader.get_metadata()
        assert metadata.get("version") == "dummy"

    def test_model_loader_ensemble_vs_stack(self):
        """Test different model name patterns."""
        ensemble_loader = ModelLoader(model_name="ensemble")
        stack_loader = ModelLoader(model_name="A")  # Stack A

        assert ensemble_loader.model_name == "ensemble"
        assert stack_loader.model_name == "A"

        # Both should be loaded
        assert ensemble_loader.is_loaded()
        assert stack_loader.is_loaded()
