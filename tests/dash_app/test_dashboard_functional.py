"""Simplified functional tests for dashboard components."""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest

from dash_app.dashboard.app import PersonalityClassifierApp, create_app
from dash_app.dashboard.layout import create_layout, create_professional_header
from dash_app.dashboard.model_loader import ModelLoader


class TestDashboardFunctionality:
    """Test the actual dashboard functionality."""

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_initialization(self, mock_load_model):
        """Test that the app initializes correctly."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(model_name="test_model")

        assert app.model_name == "test_model"
        assert app.host == "127.0.0.1"
        assert app.port == 8050
        assert app.app is not None

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_with_custom_params(self, mock_load_model):
        """Test app with custom parameters."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(
            model_name="custom_model",
            model_version="v1.0",
            host="0.0.0.0",
            port=9000
        )

        assert app.model_name == "custom_model"
        assert app.model_version == "v1.0"
        assert app.host == "0.0.0.0"
        assert app.port == 9000

    def test_create_app_function(self):
        """Test the create_app factory function."""
        with patch('dash_app.dashboard.app.PersonalityClassifierApp') as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            create_app("test_model")

            mock_app.assert_called_once_with(
                model_name="test_model",
                model_version=None,
                model_stage="Production"
            )

    def test_layout_creation(self):
        """Test layout creation."""
        model_name = "test_model"
        model_metadata = {"version": "1.0"}

        layout = create_layout(model_name, model_metadata)

        assert layout is not None

    def test_professional_header_creation(self):
        """Test professional header creation."""
        header = create_professional_header()

        # The function returns a dbc.Container, not html.Div
        assert isinstance(header, dbc.Container)

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_model_loader_initialization(self, mock_load_model):
        """Test model loader initialization."""
        mock_load_model.return_value = None

        loader = ModelLoader("test_model")

        assert loader.model_name == "test_model"
        assert loader.model_stage == "Production"

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_has_prediction_history(self, mock_load_model):
        """Test that app has prediction history."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(model_name="test_model")

        assert hasattr(app, 'prediction_history')
        assert isinstance(app.prediction_history, list)

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_has_callback_registration(self, mock_load_model):
        """Test that callbacks are registered."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(model_name="test_model")

        # Check that the app has callbacks registered
        assert hasattr(app.app, 'callback_map')

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_run_method(self, mock_load_model):
        """Test app run method."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(model_name="test_model")
        app.app.run_server = MagicMock()

        app.run(debug=True)

        app.app.run_server.assert_called_once_with(
            host="127.0.0.1",
            port=8050,
            debug=True
        )

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_get_app_method(self, mock_load_model):
        """Test get_app method."""
        mock_load_model.return_value = None

        app = PersonalityClassifierApp(model_name="test_model")
        dash_app = app.get_app()

        assert dash_app is app.app


class TestModelLoaderFunctionality:
    """Test model loader functionality."""

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_model_loader_attributes(self, mock_load_model):
        """Test model loader has correct attributes."""
        mock_load_model.return_value = None

        loader = ModelLoader("test_model", "v1.0", "Staging")

        assert loader.model_name == "test_model"
        assert loader.model_version == "v1.0"
        assert loader.model_stage == "Staging"

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_model_loader_has_model_attribute(self, mock_load_model):
        """Test that model loader has model attribute."""
        mock_load_model.return_value = None

        loader = ModelLoader("test_model")

        assert hasattr(loader, 'model')

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_model_loader_has_metadata(self, mock_load_model):
        """Test that model loader has metadata."""
        mock_load_model.return_value = None

        loader = ModelLoader("test_model")

        assert hasattr(loader, 'model_metadata')
        assert isinstance(loader.model_metadata, dict)


class TestIntegrationWorkflow:
    """Test integration workflow."""

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_complete_app_creation_workflow(self, mock_load_model):
        """Test complete app creation workflow."""
        mock_load_model.return_value = None

        # Create app
        app = PersonalityClassifierApp(model_name="ensemble_model")

        # Verify all components are set up
        assert app.model_name == "ensemble_model"
        assert app.app is not None
        assert app.app.layout is not None
        assert app.model_loader is not None
        assert isinstance(app.prediction_history, list)

    @patch('dash_app.dashboard.model_loader.ModelLoader._load_model')
    def test_app_scalability(self, mock_load_model):
        """Test that multiple apps can be created."""
        mock_load_model.return_value = None

        apps = []
        for i in range(3):
            app = PersonalityClassifierApp(model_name=f"model_{i}")
            apps.append(app)

        assert len(apps) == 3
        for app in apps:
            assert app.app is not None


if __name__ == "__main__":
    pytest.main([__file__])
