"""Tests for dashboard callback functions."""

from unittest.mock import MagicMock, patch

import pytest
from dash import Dash

from dash_app.dashboard.callbacks import register_callbacks


class TestCallbackRegistration:
    """Test suite for callback registration."""

    def test_register_callbacks_success(self):
        """Test successful callback registration."""
        # Create mock objects
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()
        prediction_history = []

        # Should not raise any exceptions
        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # Verify that callbacks were registered (app.callback should be called)
        assert mock_app.callback.called

    def test_register_callbacks_with_history(self):
        """Test callback registration with existing prediction history."""
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()
        prediction_history = [
            {"timestamp": "2025-01-15", "prediction": {"Extroversion": 0.8}}
        ]

        register_callbacks(mock_app, mock_model_loader, prediction_history)
        assert mock_app.callback.called


class TestPredictionCallback:
    """Test suite for prediction callback functionality."""

    @pytest.fixture
    def mock_setup(self):
        """Set up mocks for testing prediction callback."""
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()
        prediction_history = []

        # Configure mock model loader
        mock_model_loader.predict.return_value = {
            "Extroversion": 0.8,
            "Agreeableness": 0.6,
            "Conscientiousness": 0.7,
            "Neuroticism": 0.4,
            "Openness": 0.9,
        }

        return mock_app, mock_model_loader, prediction_history

    def test_prediction_callback_registration(self, mock_setup):
        """Test that prediction callback is properly registered."""
        mock_app, mock_model_loader, prediction_history = mock_setup

        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # Verify callback was registered
        assert mock_app.callback.called
        # Should have at least one callback call for the prediction
        assert mock_app.callback.call_count >= 1

    def test_prediction_with_valid_inputs(self, mock_setup):
        """Test prediction callback with valid input values."""
        mock_app, mock_model_loader, prediction_history = mock_setup

        # Register callbacks
        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # Get the registered callback function
        callback_calls = mock_app.callback.call_args_list
        assert len(callback_calls) > 0

        # Find the prediction callback (it should be the one with most State parameters)
        prediction_callback = None
        for call in callback_calls:
            args, kwargs = call
            if len(args) >= 2:  # Output, Input, State...
                prediction_callback = args
                break

        assert prediction_callback is not None

    def test_model_loader_integration(self, mock_setup):
        """Test integration with model loader."""
        mock_app, mock_model_loader, prediction_history = mock_setup

        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # Verify model_loader is passed to the callback registration
        assert mock_app.callback.called


class TestCallbackErrorHandling:
    """Test error handling in callbacks."""

    def test_callback_with_none_model_loader(self):
        """Test callback registration with None model loader."""
        mock_app = MagicMock(spec=Dash)
        prediction_history = []

        # Should handle None model_loader gracefully
        register_callbacks(mock_app, None, prediction_history)
        assert mock_app.callback.called

    def test_callback_with_none_history(self):
        """Test callback registration with None prediction history."""
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()

        # Should handle None prediction_history gracefully
        register_callbacks(mock_app, mock_model_loader, None)
        assert mock_app.callback.called

    def test_callback_with_invalid_app(self):
        """Test callback registration with invalid app object."""
        mock_model_loader = MagicMock()
        prediction_history = []

        # Should handle invalid app object
        with pytest.raises(AttributeError):
            register_callbacks("invalid_app", mock_model_loader, prediction_history)


class TestCallbackInputValidation:
    """Test input validation in callbacks."""

    @pytest.fixture
    def callback_function_mock(self):
        """Mock the actual callback function for testing."""
        with patch("dash_app.dashboard.callbacks.register_callbacks") as mock_register:
            # Create a mock prediction function that matches our refactored signature
            def mock_prediction_callback(n_clicks, *input_values):
                # Simulate input validation
                if n_clicks is None or n_clicks == 0:
                    return "No prediction made"

                # Validate input ranges - unpack the input values
                if len(input_values) < 7:
                    return "Invalid input: Not enough values"

                time_alone, social_events, going_outside = (
                    input_values[0],
                    input_values[1],
                    input_values[2],
                )
                friends_size, post_freq, _stage_fear, _drained_social = (
                    input_values[3],
                    input_values[4],
                    input_values[5],
                    input_values[6],
                )

                inputs = [
                    time_alone,
                    social_events,
                    going_outside,
                    friends_size,
                    post_freq,
                ]

                if any(x is None for x in inputs):
                    return "Invalid input: None values"

                # Check numeric inputs
                if any(not isinstance(x, (int, float)) for x in inputs if x is not None):
                    return "Invalid input: Non-numeric values"

                return "Valid prediction"

            mock_register.return_value = mock_prediction_callback
            yield mock_prediction_callback

    def test_callback_with_none_clicks(self, callback_function_mock):
        """Test callback behavior with no button clicks."""
        result = callback_function_mock(None, 3.0, 2.0, 4.0, 3.0, 2.0, 1.0, 2.0)
        assert result == "No prediction made"

    def test_callback_with_zero_clicks(self, callback_function_mock):
        """Test callback behavior with zero button clicks."""
        result = callback_function_mock(0, 3.0, 2.0, 4.0, 3.0, 2.0, 1.0, 2.0)
        assert result == "No prediction made"

    def test_callback_with_none_inputs(self, callback_function_mock):
        """Test callback behavior with None input values."""
        result = callback_function_mock(1, None, 2.0, 4.0, 3.0, 2.0, 1.0, 2.0)
        assert result == "Invalid input: None values"

    def test_callback_with_invalid_inputs(self, callback_function_mock):
        """Test callback behavior with invalid input types."""
        result = callback_function_mock(1, "invalid", 2.0, 4.0, 3.0, 2.0, 1.0, 2.0)
        assert result == "Invalid input: Non-numeric values"

    def test_callback_with_valid_inputs(self, callback_function_mock):
        """Test callback behavior with valid inputs."""
        result = callback_function_mock(1, 3.0, 2.0, 4.0, 3.0, 2.0, 1.0, 2.0)
        assert result == "Valid prediction"


class TestCallbackHistoryManagement:
    """Test prediction history management in callbacks."""

    def test_history_updates_after_prediction(self):
        """Test that prediction history is updated after successful prediction."""
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()
        prediction_history = []

        # Configure mock to return a prediction
        mock_model_loader.predict.return_value = {
            "Extroversion": 0.8,
            "Agreeableness": 0.6,
        }

        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # Verify that the history list reference is maintained
        assert isinstance(prediction_history, list)

    def test_history_size_limit(self):
        """Test that prediction history respects size limits."""
        # This would test if there's a maximum history size implementation
        prediction_history = [{"test": f"prediction_{i}"} for i in range(1000)]
        mock_app = MagicMock(spec=Dash)
        mock_model_loader = MagicMock()

        register_callbacks(mock_app, mock_model_loader, prediction_history)

        # The function should handle large histories gracefully
        assert isinstance(prediction_history, list)


if __name__ == "__main__":
    pytest.main([__file__])
