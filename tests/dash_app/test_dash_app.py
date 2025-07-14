"""Tests for the Dash application."""

import pytest


@pytest.mark.integration
def test_dash_app_creation(dash_app):
    """Test that the Dash app can be created."""
    assert dash_app is not None
    app = dash_app.get_app()
    assert app is not None


@pytest.mark.integration
def test_dash_app_layout(dash_app):
    """Test that the Dash app has a valid layout."""
    app = dash_app.get_app()
    assert app.layout is not None


@pytest.mark.integration
def test_dash_app_server(dash_client):
    """Test that the Dash app server responds."""
    response = dash_client.get('/')
    assert response.status_code == 200
