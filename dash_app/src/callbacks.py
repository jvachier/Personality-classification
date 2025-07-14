"""Callback functions for the Dash application."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from dash import dash_table, html
from dash.dependencies import Input, Output, State

from .layout import (
    create_file_input,
    create_json_input,
    create_manual_input,
    format_prediction_result,
)


def register_callbacks(app, model_loader, prediction_history: list) -> None:
    """Register all callbacks for the Dash application.

    Args:
        app: Dash application instance
        model_loader: Model loader instance
        prediction_history: List to store prediction history
    """
    logger = logging.getLogger(__name__)

    @app.callback(Output("input-content", "children"), Input("input-tabs", "value"))
    def update_input_content(tab_value):
        """Update input content based on selected tab."""
        if tab_value == "manual":
            return create_manual_input()
        elif tab_value == "json":
            return create_json_input()
        elif tab_value == "file":
            return create_file_input()
        return html.Div("Select an input method")

    @app.callback(
        Output("prediction-results", "children"),
        Input("predict-button", "n_clicks"),
        State("input-tabs", "value"),
        State("feature1", "value"),
        State("feature2", "value"),
        State("feature3", "value"),
        State("json-input", "value"),
        prevent_initial_call=True,
    )
    def make_prediction(n_clicks, input_type, feature1, feature2, feature3, json_input):
        """Handle prediction requests."""
        if not n_clicks:
            return ""

        try:
            # Extract input data based on input type
            if input_type == "manual":
                data = {
                    "feature1": feature1 if feature1 is not None else 0.5,
                    "feature2": feature2 if feature2 is not None else 0.3,
                    "feature3": feature3 if feature3 is not None else 0.8,
                }
            elif input_type == "json":
                if not json_input:
                    return html.Div("Please provide JSON input", style={"color": "red"})
                try:
                    data = json.loads(json_input)
                except json.JSONDecodeError:
                    return html.Div("Invalid JSON format", style={"color": "red"})
            else:
                return html.Div(
                    "File upload not implemented yet", style={"color": "orange"}
                )

            # Make prediction
            result = model_loader.predict(data)
            result["timestamp"] = datetime.now().isoformat()

            # Add to history
            prediction_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "input": data,
                    "result": result,
                }
            )

            return format_prediction_result(result)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return html.Div(f"Error: {e!s}", style={"color": "red"})

    @app.callback(
        Output("prediction-history", "children"),
        Input("interval-component", "n_intervals"),
        Input("predict-button", "n_clicks"),
    )
    def update_prediction_history(n_intervals, n_clicks):
        """Update prediction history display."""
        if not prediction_history:
            return html.Div("No predictions yet", style={"color": "#7f8c8d"})

        # Create table data
        table_data = []
        for i, pred in enumerate(reversed(prediction_history[-10:])):  # Show last 10
            table_data.append(
                {
                    "ID": f"#{len(prediction_history) - i}",
                    "Timestamp": pred["timestamp"][:19],  # Remove microseconds
                    "Prediction": pred["result"].get("prediction", "N/A"),
                    "Confidence": f"{pred['result'].get('confidence', 0):.3f}"
                    if pred["result"].get("confidence")
                    else "N/A",
                }
            )

        return dash_table.DataTable(
            data=table_data,
            columns=[
                {"name": "ID", "id": "ID"},
                {"name": "Timestamp", "id": "Timestamp"},
                {"name": "Prediction", "id": "Prediction"},
                {"name": "Confidence", "id": "Confidence"},
            ],
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={
                "backgroundColor": "#3498db",
                "color": "white",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": 0},
                    "backgroundColor": "#ecf0f1",
                }
            ],
        )

    @app.callback(
        Output("interval-component", "disabled"), Input("auto-refresh-toggle", "value")
    )
    def toggle_auto_refresh(value):
        """Toggle auto-refresh based on checkbox."""
        return "auto" not in value

    @app.callback(
        Output("json-input", "value"),
        Input("json-input-display", "value"),
        prevent_initial_call=True,
    )
    def sync_json_input(value):
        """Sync the display JSON input with the hidden one."""
        return value
