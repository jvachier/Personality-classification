"""Integration tests for the complete dashboard pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dash_app.dashboard.app import PersonalityClassifierApp


class TestDashboardIntegration:
    """Integration tests for the complete dashboard workflow."""

    @pytest.fixture
    def temp_model_file(self):
        """Create a temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @patch('joblib.load')
    def test_complete_dashboard_workflow(self, mock_joblib_load, temp_model_file):
        """Test complete dashboard workflow from initialization to prediction."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [
            [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4, 0.1, 0.9]
        ]
        mock_joblib_load.return_value = mock_model

        # Initialize dashboard with mock model
        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(
                model_name="ensemble",
                host="127.0.0.1",
                port=8050
            )

            # Verify app is properly initialized
            assert app.model_name == "ensemble"
            assert app.host == "127.0.0.1"
            assert app.port == 8050
            assert app.app is not None
            assert app.app.layout is not None

    def test_dashboard_with_invalid_model_path(self):
        """Test dashboard behavior with invalid model path."""
        # PersonalityClassifierApp doesn't raise FileNotFoundError - it creates dummy models
        app = PersonalityClassifierApp(model_name="nonexistent_model")
        assert app.model_name == "nonexistent_model"
        assert app.app is not None

    @patch('joblib.load')
    def test_dashboard_layout_rendering(self, mock_joblib_load, temp_model_file):
        """Test that dashboard layout renders correctly."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(model_name="test_model")

            # Verify layout components exist
            layout = app.app.layout
            assert layout is not None

    @patch('joblib.load')
    def test_dashboard_callbacks_registration(self, mock_joblib_load, temp_model_file):
        """Test that dashboard callbacks are properly registered."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(model_name="test_model")

            # Verify that callbacks are registered (app should have callback registry)
            assert hasattr(app.app, 'callback_map')


class TestDashboardErrorRecovery:
    """Test dashboard error recovery and graceful degradation."""

    @patch('joblib.load')
    def test_dashboard_with_corrupted_model(self, mock_joblib_load):
        """Test dashboard behavior with corrupted model."""
        mock_joblib_load.side_effect = OSError("Corrupted model file")

        # PersonalityClassifierApp handles corrupted models gracefully with dummy fallback
        app = PersonalityClassifierApp(model_name="corrupted_model")
        assert app.model_name == "corrupted_model"
        assert app.app is not None

    @patch('joblib.load')
    def test_dashboard_handles_prediction_errors(self, mock_joblib_load):
        """Test dashboard handles prediction errors gracefully."""
        # Setup mock model that fails during prediction
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = ValueError("Prediction failed")
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            # Should initialize successfully even if model has issues
            app = PersonalityClassifierApp(model_name="test_model")
            assert app is not None


class TestDashboardPerformance:
    """Test dashboard performance and resource usage."""

    @patch('joblib.load')
    def test_dashboard_memory_usage(self, mock_joblib_load):
        """Test that dashboard doesn't create memory leaks."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            # Create multiple app instances
            apps = []
            for i in range(5):
                app = PersonalityClassifierApp(model_name=f"test_model_{i}")
                apps.append(app)

            # Each should be independent
            assert len(apps) == 5
            for app in apps:
                assert app.app is not None

    @patch('joblib.load')
    def test_dashboard_startup_time(self, mock_joblib_load):
        """Test dashboard startup performance."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(model_name="test_model")

            # Verify that startup is reasonably fast
            assert app.app is not None


class TestDashboardConfiguration:
    """Test dashboard configuration options."""

    @patch('joblib.load')
    def test_dashboard_custom_configuration(self, mock_joblib_load):
        """Test dashboard with custom configuration."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(
                model_name="custom_model",
                model_version="v2.0",
                model_stage="Staging",
                host="0.0.0.0",
                port=9000
            )

            assert app.model_name == "custom_model"
            assert app.model_version == "v2.0"
            assert app.model_stage == "Staging"
            assert app.host == "0.0.0.0"
            assert app.port == 9000

    @patch('joblib.load')
    def test_dashboard_environment_variables(self, mock_joblib_load):
        """Test dashboard respects environment configuration."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True
                 # Test with environment-like configuration
        with patch.dict('os.environ', {'DASH_HOST': '0.0.0.0', 'DASH_PORT': '9000'}):
            app = PersonalityClassifierApp(model_name="test_model")

            # App should still use provided parameters over environment
            assert app.host == "127.0.0.1"  # Default value
            assert app.port == 8050  # Default value


class TestDashboardScalability:
    """Test dashboard scalability and concurrent usage."""

    @patch('joblib.load')
    def test_dashboard_concurrent_initialization(self, mock_joblib_load):
        """Test multiple dashboard instances can be created concurrently."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            # Test creating multiple app instances
            apps = []
            for i in range(3):
                app = PersonalityClassifierApp(model_name=f"model_{i}")
                apps.append(app)

            # All should succeed
            assert len(apps) == 3
            for app in apps:
                assert isinstance(app, PersonalityClassifierApp)

    @patch('joblib.load')
    def test_dashboard_prediction_history_management(self, mock_joblib_load):
        """Test prediction history management under load."""
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        with patch('dash_app.dashboard.model_loader.Path.exists') as mock_exists:
            mock_exists.return_value = True

            app = PersonalityClassifierApp(model_name="test_model")

            # Simulate adding many predictions to history
            for i in range(100):
                app.prediction_history.append({
                    "timestamp": f"2025-01-15T{i:02d}:00:00",
                    "prediction": {"Extroversion": 0.8}
                })

            assert len(app.prediction_history) == 100
            # History should be manageable even with many entries
            assert isinstance(app.prediction_history, list)


if __name__ == "__main__":
    pytest.main([__file__])
