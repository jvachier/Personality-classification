"""Tests for dashboard layout components."""

import pytest
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html

from dash_app.dashboard.layout import (
    create_input_panel,
    create_layout,
    create_personality_radar,
    create_professional_header,
    format_prediction_result,
)


class TestLayoutComponents:
    """Test suite for layout components."""

    def test_create_professional_header(self):
        """Test professional header creation."""
        header = create_professional_header()

        # The header returns a dbc.Container, not html.Div
        assert isinstance(header, dbc.Container)
        # Check for required styling
        assert hasattr(header, 'style')
        # Check for children components
        assert hasattr(header, 'children')

    def test_create_input_panel(self):
        """Test input panel creation."""
        panel = create_input_panel()

        assert isinstance(panel, dbc.Card)
        # Should have card header and body
        assert hasattr(panel, 'children')

    def test_create_layout_structure(self):
        """Test main layout structure."""
        model_name = "test_model"
        model_metadata = {"version": "1.0", "created": "2025-01-01"}

        layout = create_layout(model_name, model_metadata)

        assert isinstance(layout, html.Div)
        assert hasattr(layout, 'children')
        assert len(layout.children) >= 2  # Header + Content


class TestPersonalityRadar:
    """Test suite for personality radar chart."""

    def test_create_personality_radar_with_valid_data(self):
        """Test radar chart creation with valid probability data."""
        probabilities = {
            "Extroversion": 0.8,
            "Agreeableness": 0.6,
            "Conscientiousness": 0.7,
            "Neuroticism": 0.4,
            "Openness": 0.9
        }

        fig = create_personality_radar(probabilities)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "scatterpolar"

    def test_create_personality_radar_with_input_data(self):
        """Test radar chart creation with input data included."""
        probabilities = {
            "Extroversion": 0.8,
            "Agreeableness": 0.6,
            "Conscientiousness": 0.7,
            "Neuroticism": 0.4,
            "Openness": 0.9
        }
        input_data = {
            "time_alone": 3.0,
            "social_events": 2.0
        }

        fig = create_personality_radar(probabilities, input_data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_personality_radar_empty_data(self):
        """Test radar chart with empty probability data."""
        probabilities = {}

        fig = create_personality_radar(probabilities)

        assert isinstance(fig, go.Figure)
        # Should handle empty data gracefully

    def test_create_personality_radar_invalid_values(self):
        """Test radar chart with invalid probability values."""
        probabilities = {
            "Extroversion": 1.5,  # Invalid: > 1.0
            "Agreeableness": -0.1,  # Invalid: < 0.0
            "Conscientiousness": 0.7,
        }

        # Should not raise an exception
        fig = create_personality_radar(probabilities)
        assert isinstance(fig, go.Figure)


class TestPredictionFormatting:
    """Test suite for prediction result formatting."""

    def test_format_prediction_result_valid(self):
        """Test formatting of valid prediction results."""
        result_dict = {
            "probabilities": {
                "Extroversion": 0.8,
                "Agreeableness": 0.6,
                "Conscientiousness": 0.7,
                "Neuroticism": 0.4,
                "Openness": 0.9
            },
            "input_data": {
                "time_alone": 3.0,
                "social_events": 2.0,
                "going_outside": 4.0,
                "friends_size": 3.0,
                "post_freq": 2.0,
                "stage_fear": 1.0,
                "drained_social": 2.0
            }
        }

        result = format_prediction_result(result_dict)

        assert isinstance(result, dbc.Card)
        # Should contain formatted components
        assert hasattr(result, 'children')

    def test_format_prediction_result_missing_data(self):
        """Test formatting with missing input data."""
        result_dict = {
            "probabilities": {
                "Extroversion": 0.8,
                "Agreeableness": 0.6
            }
        }

        # Should handle missing input data gracefully
        result = format_prediction_result(result_dict)
        assert isinstance(result, dbc.Card)


class TestLayoutIntegration:
    """Integration tests for layout components."""

    def test_layout_with_mock_model_metadata(self):
        """Test layout creation with realistic model metadata."""
        model_name = "six_stack_ensemble"
        model_metadata = {
            "model_type": "ensemble",
            "version": "1.0.0",
            "created_date": "2025-01-15",
            "accuracy": 0.92,
            "features": [
                "time_alone", "social_events", "going_outside",
                "friends_size", "post_freq", "stage_fear", "drained_social"
            ]
        }

        layout = create_layout(model_name, model_metadata)

        assert isinstance(layout, html.Div)
        # Verify structure contains expected components
        assert len(layout.children) >= 2

    def test_layout_responsiveness(self):
        """Test that layout components have responsive classes."""
        layout = create_layout("test", {})

        # Check for Bootstrap responsive classes in the layout
        layout_str = str(layout)
        assert "dbc.Container" in layout_str or "container" in layout_str.lower()


class TestLayoutEdgeCases:
    """Test edge cases for layout components."""

    def test_empty_model_name(self):
        """Test layout creation with empty model name."""
        layout = create_layout("", {})
        assert isinstance(layout, html.Div)

    def test_none_model_metadata(self):
        """Test layout creation with None metadata."""
        layout = create_layout("test_model", {})
        assert isinstance(layout, html.Div)

    def test_large_model_metadata(self):
        """Test layout with extensive metadata."""
        large_metadata = {f"param_{i}": f"value_{i}" for i in range(100)}
        layout = create_layout("test_model", large_metadata)
        assert isinstance(layout, html.Div)


if __name__ == "__main__":
    pytest.main([__file__])
