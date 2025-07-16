"""Callback functions for the Dash application."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from dash import html
from dash.dependencies import Input, Output, State

from .layout import (
    format_prediction_result,
)


@dataclass
class PredictionInputs:
    """Data class for prediction input parameters."""

    time_alone: float | None = None
    social_events: float | None = None
    going_outside: float | None = None
    friends_size: float | None = None
    post_freq: float | None = None
    stage_fear: str | None = None
    drained_social: str | None = None

    def to_feature_dict(self) -> dict[str, float]:
        """Convert inputs to feature dictionary for model prediction."""
        return {
            "Time_spent_Alone": self.time_alone if self.time_alone is not None else 2.0,
            "Social_event_attendance": self.social_events
            if self.social_events is not None
            else 4.0,
            "Going_outside": self.going_outside
            if self.going_outside is not None
            else 3.0,
            "Friends_circle_size": self.friends_size
            if self.friends_size is not None
            else 8.0,
            "Post_frequency": self.post_freq if self.post_freq is not None else 3.0,
            # One-hot encode Stage_fear
            "Stage_fear_No": 1 if self.stage_fear == "No" else 0,
            "Stage_fear_Unknown": 1 if self.stage_fear == "Unknown" else 0,
            "Stage_fear_Yes": 1 if self.stage_fear == "Yes" else 0,
            # One-hot encode Drained_after_socializing
            "Drained_after_socializing_No": 1 if self.drained_social == "No" else 0,
            "Drained_after_socializing_Unknown": 1
            if self.drained_social == "Unknown"
            else 0,
            "Drained_after_socializing_Yes": 1 if self.drained_social == "Yes" else 0,
            # Set external match features to Unknown (default)
            "match_p_Extrovert": 0,
            "match_p_Introvert": 0,
            "match_p_Unknown": 1,
        }


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
    def make_prediction(n_clicks, *input_values):
        """Handle prediction requests."""
        if not n_clicks:
            return ""

        try:
            # Create prediction inputs from callback arguments
            inputs = PredictionInputs(
                time_alone=input_values[0],
                social_events=input_values[1],
                going_outside=input_values[2],
                friends_size=input_values[3],
                post_freq=input_values[4],
                stage_fear=input_values[5],
                drained_social=input_values[6],
            )

            # Convert to feature dictionary
            data = inputs.to_feature_dict()

            # Make prediction
            result = model_loader.predict(data)
            result["timestamp"] = datetime.now().isoformat()
            result["input_data"] = data  # Add input data for radar chart

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

    # Enhanced predict button with loading states
    @app.callback(
        [
            Output("predict-button", "children"),
            Output("predict-button", "disabled"),
            Output("predict-button", "color"),
        ],
        [Input("predict-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def update_predict_button(n_clicks):
        """Update predict button state with loading animation."""
        if n_clicks:
            # Show loading state briefly (will be overridden by prediction callback)
            return [
                [
                    html.I(className="fas fa-spinner fa-spin me-2"),
                    "Analyzing Your Personality...",
                ],
                True,
                "warning",
            ]

        # Default state
        return [
            [html.I(className="fas fa-magic me-2"), "Analyze My Personality"],
            False,
            "primary",
        ]

    # Reset button state after prediction
    @app.callback(
        [
            Output("predict-button", "children", allow_duplicate=True),
            Output("predict-button", "disabled", allow_duplicate=True),
            Output("predict-button", "color", allow_duplicate=True),
        ],
        [Input("prediction-results", "children")],
        prevent_initial_call=True,
    )
    def reset_predict_button(results):
        """Reset predict button after prediction is complete."""
        if results:
            return [
                [html.I(className="fas fa-magic me-2"), "Analyze Again"],
                False,
                "success",
            ]

        return [
            [html.I(className="fas fa-magic me-2"), "Analyze My Personality"],
            False,
            "primary",
        ]
