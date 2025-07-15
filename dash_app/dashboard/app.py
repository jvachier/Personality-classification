"""Main Dash application for personality classification model serving."""

from __future__ import annotations

import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc

from .callbacks import register_callbacks
from .layout import create_layout
from .model_loader import ModelLoader


class PersonalityClassifierApp:
    """Main Dash application for personality classification."""

    def __init__(
        self,
        model_name: str,
        model_version: str | None = None,
        model_stage: str = "Production",
        host: str = "127.0.0.1",
        port: int = 8050,
    ):
        """Initialize the Dash application.

        Args:
            model_name: Name of the model to serve
            model_version: Specific version to serve (optional)
            model_stage: Stage to serve from if version not specified
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_stage = model_stage
        self.host = host
        self.port = port

        self.logger = logging.getLogger(__name__)
        self.prediction_history: list[dict[str, Any]] = []

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            title=f"Personality Classifier - {model_name}",
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        )

        # Add custom CSS to ensure white background
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        background-color: #ffffff !important;
                        margin: 0;
                        padding: 0;
                    }
                    html {
                        background-color: #ffffff !important;
                    }
                    .dash-bootstrap, .container-fluid, .container {
                        background-color: #ffffff !important;
                    }
                    ._dash-loading {
                        background-color: #ffffff !important;
                    }
                    #react-entry-point {
                        background-color: #ffffff !important;
                    }
                </style>
            </head>
            <body style="background-color: #ffffff !important; margin: 0; padding: 0;">
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

        # Load model
        self.model_loader = ModelLoader(model_name, model_version, model_stage)

        # Setup layout and callbacks
        self.app.layout = create_layout(
            self.model_name, self.model_loader.get_metadata()
        )
        register_callbacks(self.app, self.model_loader, self.prediction_history)

    def run(self, debug: bool = False) -> None:
        """Run the Dash server.

        Args:
            debug: Whether to run in debug mode
        """
        self.logger.info(f"Starting Dash model server on {self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=debug)

    def get_app(self) -> dash.Dash:
        """Get the Dash app instance.

        Returns:
            The Dash application instance
        """
        return self.app


def create_app(
    model_name: str,
    model_version: str | None = None,
    model_stage: str = "Production",
) -> dash.Dash:
    """Factory function to create a Dash app.

    Args:
        model_name: Name of the model to serve
        model_version: Specific version to serve
        model_stage: Stage to serve from

    Returns:
        Dash application instance
    """
    app_instance = PersonalityClassifierApp(
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
    )
    return app_instance.get_app()
