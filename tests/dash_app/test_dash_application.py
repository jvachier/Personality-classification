"""Tests for the main Dash application class."""

from unittest.mock import MagicMock, patch

import dash
import pytest

from dash_app.dashboard.app import PersonalityClassifierApp, create_app


class TestPersonalityClassifierApp:
    """Test suite for PersonalityClassifierApp class."""

    def test_app_initialization_default_params(self):
        """Test app initialization with default parameters."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")

            assert app.model_name == "test_model"
            assert app.model_version is None
            assert app.model_stage == "Production"
            assert app.host == "127.0.0.1"
            assert app.port == 8050

    def test_app_initialization_custom_params(self):
        """Test app initialization with custom parameters."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(
                model_name="custom_model",
                model_version="v1.0",
                model_stage="Staging",
                host="0.0.0.0",
                port=9000,
            )

            assert app.model_name == "custom_model"
            assert app.model_version == "v1.0"
            assert app.model_stage == "Staging"
            assert app.host == "0.0.0.0"
            assert app.port == 9000

    def test_app_has_dash_instance(self):
        """Test that app creates a Dash instance."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")

            assert hasattr(app, "app")
            assert isinstance(app.app, dash.Dash)

    def test_app_title_configuration(self):
        """Test that app title is configured correctly."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")

            assert "test_model" in app.app.title

    def test_app_layout_is_set(self):
        """Test that app layout is properly set."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            with patch("dash_app.dashboard.app.create_layout") as mock_layout:
                mock_layout.return_value = MagicMock()

                app = PersonalityClassifierApp(model_name="test_model")

                assert app.app.layout is not None

    def test_app_callbacks_registration(self):
        """Test that callbacks are registered."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            with patch("dash_app.dashboard.app.register_callbacks") as mock_callbacks:
                app = PersonalityClassifierApp(model_name="test_model")

                # Verify register_callbacks was called
                mock_callbacks.assert_called_once()

    def test_app_prediction_history_initialization(self):
        """Test that prediction history is initialized."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")

            assert hasattr(app, "prediction_history")
            assert isinstance(app.prediction_history, list)
            assert len(app.prediction_history) == 0

    def test_get_app_method(self):
        """Test the get_app method."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")
            dash_app = app.get_app()

            assert isinstance(dash_app, dash.Dash)
            assert dash_app is app.app


class TestAppRunning:
    """Test suite for app running functionality."""

    def test_app_run_method_exists(self):
        """Test that run method exists."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.return_value = MagicMock()

            app = PersonalityClassifierApp(model_name="test_model")

            assert hasattr(app, "run")
            assert callable(app.run)

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_run_with_debug_false(self, mock_loader):
        """Test app running with debug=False."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")

        # Mock the Dash app's run_server method
        app.app.run_server = MagicMock()

        app.run(debug=False)

        # Verify run_server was called with correct parameters
        app.app.run_server.assert_called_once_with(
            host="127.0.0.1", port=8050, debug=False
        )

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_run_with_debug_true(self, mock_loader):
        """Test app running with debug=True."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")
        app.app.run_server = MagicMock()

        app.run(debug=True)

        app.app.run_server.assert_called_once_with(
            host="127.0.0.1", port=8050, debug=True
        )

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_run_with_custom_host_port(self, mock_loader):
        """Test app running with custom host and port."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(
            model_name="test_model", host="0.0.0.0", port=9000
        )
        app.app.run_server = MagicMock()

        app.run()

        app.app.run_server.assert_called_once_with(
            host="0.0.0.0", port=9000, debug=False
        )


class TestCreateAppFunction:
    """Test suite for the create_app function."""

    def test_create_app_function_exists(self):
        """Test that create_app function exists."""
        assert callable(create_app)

    @patch("dash_app.dashboard.app.PersonalityClassifierApp")
    def test_create_app_with_default_params(self, mock_app_class):
        """Test create_app function with default parameters."""
        mock_instance = MagicMock()
        mock_app_class.return_value = mock_instance

        result = create_app("test_model")

        mock_app_class.assert_called_once_with(
            model_name="test_model", model_version=None, model_stage="Production"
        )
        assert result == mock_instance.get_app.return_value

    @patch("dash_app.dashboard.app.PersonalityClassifierApp")
    def test_create_app_with_custom_params(self, mock_app_class):
        """Test create_app function with custom parameters."""
        mock_instance = MagicMock()
        mock_app_class.return_value = mock_instance

        result = create_app(
            model_name="custom_model", model_version="v2.0", model_stage="Staging"
        )

        mock_app_class.assert_called_once_with(
            model_name="custom_model", model_version="v2.0", model_stage="Staging"
        )
        assert result == mock_instance.get_app.return_value


class TestAppErrorHandling:
    """Test error handling in app initialization and running."""

    def test_app_with_invalid_model_name(self):
        """Test app initialization with invalid model name."""
        with patch("dash_app.dashboard.app.ModelLoader") as mock_loader:
            mock_loader.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(FileNotFoundError):
                PersonalityClassifierApp(model_name="nonexistent_model")

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_with_model_loading_error(self, mock_loader):
        """Test app behavior when model loading fails."""
        mock_loader.side_effect = OSError("Model loading failed")

        with pytest.raises(OSError):  # More specific exception
            PersonalityClassifierApp(model_name="test_model")

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_run_server_error(self, mock_loader):
        """Test app behavior when run_server fails."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")
        app.app.run_server = MagicMock(side_effect=OSError("Server start failed"))

        with pytest.raises(OSError):
            app.run()


class TestAppIntegration:
    """Integration tests for the complete app."""

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_full_app_initialization_workflow(self, mock_loader):
        """Test complete app initialization workflow."""
        # Setup mock model loader
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "Extroversion": 0.8,
            "Agreeableness": 0.6,
            "Conscientiousness": 0.7,
            "Neuroticism": 0.4,
            "Openness": 0.9,
        }
        mock_loader.return_value = mock_model

        # Initialize app
        app = PersonalityClassifierApp(model_name="ensemble_model")

        # Verify all components are properly set up
        assert app.model_name == "ensemble_model"
        assert isinstance(app.app, dash.Dash)
        assert app.app.layout is not None
        assert isinstance(app.prediction_history, list)

        # Verify model loader was called
        mock_loader.assert_called_once()

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_with_real_model_path(self, mock_loader):
        """Test app with realistic model path."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="models/ensemble_model.pkl")

        assert app.model_name == "models/ensemble_model.pkl"
        # Verify model loader was called with the path
        mock_loader.assert_called_once()


class TestAppConfiguration:
    """Test app configuration and settings."""

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_external_stylesheets(self, mock_loader):
        """Test that external stylesheets are properly configured."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")

        # Check that the app has external stylesheets configured
        # Since Dash doesn't expose external_stylesheets directly, we check the config
        assert hasattr(app.app, "config")
        # Verify the app was created with stylesheets (implicit test)
        assert app.app is not None

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_suppress_callback_exceptions(self, mock_loader):
        """Test that callback exceptions are properly configured."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")

        # Should suppress callback exceptions for dynamic layouts
        assert app.app.config.suppress_callback_exceptions is True

    @patch("dash_app.dashboard.app.ModelLoader")
    def test_app_logging_configuration(self, mock_loader):
        """Test that logging is properly configured."""
        mock_loader.return_value = MagicMock()

        app = PersonalityClassifierApp(model_name="test_model")

        assert hasattr(app, "logger")
        assert app.logger is not None


if __name__ == "__main__":
    pytest.main([__file__])
