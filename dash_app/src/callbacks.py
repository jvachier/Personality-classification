"""Callback functions for the Dash application."""

from __future__ import annotations

import logging
from datetime import datetime

from dash import dash_table, html
from dash.dependencies import Input, Output, State

from .layout import (
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

    # No longer need tab switching callback since we only have manual input

    @app.callback(
        Output("prediction-results", "children"),
        Input("predict-button", "n_clicks"),
        State("time-spent-alone", "value"),
        State("social-event-attendance", "value"),
        State("going-outside", "value"),
        State("friends-circle-size", "value"),
        State("post-frequency", "value"),
        State("stage-fear", "value"),
        State("drained-after-socializing", "value"),
        prevent_initial_call=True,
    )
    def make_prediction(
        n_clicks,
        time_alone,
        social_events,
        going_outside,
        friends_size,
        post_freq,
        stage_fear,
        drained_social,
    ):
        """Handle prediction requests."""
        if not n_clicks:
            return ""

        try:
            # Build the feature dictionary with proper encoding
            data = {
                "Time_spent_Alone": time_alone if time_alone is not None else 2.0,
                "Social_event_attendance": social_events
                if social_events is not None
                else 4.0,
                "Going_outside": going_outside if going_outside is not None else 3.0,
                "Friends_circle_size": friends_size
                if friends_size is not None
                else 8.0,
                "Post_frequency": post_freq if post_freq is not None else 3.0,
                # One-hot encode Stage_fear
                "Stage_fear_No": 1 if stage_fear == "No" else 0,
                "Stage_fear_Unknown": 1 if stage_fear == "Unknown" else 0,
                "Stage_fear_Yes": 1 if stage_fear == "Yes" else 0,
                # One-hot encode Drained_after_socializing
                "Drained_after_socializing_No": 1 if drained_social == "No" else 0,
                "Drained_after_socializing_Unknown": 1
                if drained_social == "Unknown"
                else 0,
                "Drained_after_socializing_Yes": 1 if drained_social == "Yes" else 0,
                # Set external match features to Unknown (default)
                "match_p_Extrovert": 0,
                "match_p_Introvert": 0,
                "match_p_Unknown": 1,
            }

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
