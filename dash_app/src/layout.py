"""Layout components for the Dash application."""

from __future__ import annotations

from typing import Any

from dash import dcc, html


def create_layout(model_name: str, model_metadata: dict[str, Any]) -> html.Div:
    """Create the main layout for the Dash application.

    Args:
        model_name: Name of the model
        model_metadata: Model metadata dictionary

    Returns:
        Dash HTML layout
    """
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1(
                        "Personality Classification Dashboard",
                        style={
                            "textAlign": "center",
                            "color": "#2c3e50",
                            "marginBottom": "10px",
                        },
                    ),
                    html.H3(
                        f"Model: {model_name}",
                        style={
                            "textAlign": "center",
                            "color": "#7f8c8d",
                            "marginBottom": "30px",
                        },
                    ),
                ]
            ),
            # Model Status Section
            html.Div(
                [
                    html.H3("Model Status", style={"color": "#34495e"}),
                    html.Div(
                        id="model-status",
                        children=[create_status_cards(model_metadata)],
                        style={"marginBottom": "30px"},
                    ),
                ]
            ),
            # Prediction Section
            html.Div(
                [
                    html.H3("Make Predictions", style={"color": "#34495e"}),
                    # Input methods tabs (simplified to manual only)
                    html.Div(
                        style={"display": "none"},  # Hide tabs since we only have manual input
                        children=[
                            dcc.Tabs(
                                id="input-tabs",
                                value="manual",
                                children=[
                                    dcc.Tab(label="Manual Input", value="manual"),
                                ],
                            )
                        ],
                    ),
                    # Input content (always manual input)
                    html.Div(
                        id="input-content",
                        style={"marginTop": "20px"},
                        children=[create_manual_input()],
                    ),
                    # Predict button
                    html.Div(
                        [
                            html.Button(
                                "Predict",
                                id="predict-button",
                                style={
                                    "backgroundColor": "#3498db",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "10px 20px",
                                    "fontSize": "16px",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "marginTop": "20px",
                                },
                            )
                        ],
                        style={"textAlign": "center"},
                    ),
                    # Results
                    html.Div(id="prediction-results", style={"marginTop": "30px"}),
                ],
                style={"marginBottom": "30px"},
            ),
            # Prediction History Section
            html.Div(
                [
                    html.H3("Prediction History", style={"color": "#34495e"}),
                    html.Div(id="prediction-history"),
                    # Auto-refresh toggle
                    html.Div(
                        [
                            dcc.Checklist(
                                id="auto-refresh-toggle",
                                options=[
                                    {"label": "Auto-refresh (5s)", "value": "auto"}
                                ],
                                value=[],
                                style={"marginTop": "10px"},
                            ),
                            dcc.Interval(
                                id="interval-component",
                                interval=5 * 1000,  # in milliseconds
                                n_intervals=0,
                                disabled=True,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        style={"margin": "20px", "fontFamily": "Arial, sans-serif"},
    )


def create_status_cards(model_metadata: dict[str, Any]) -> html.Div:
    """Create status cards showing model information.

    Args:
        model_metadata: Model metadata dictionary

    Returns:
        Div containing status cards
    """
    model_loaded = bool(model_metadata)
    status_color = "#27ae60" if model_loaded else "#e74c3c"
    status_text = "Loaded" if model_loaded else "Not Loaded"

    return html.Div(
        [
            # Model Status Card
            html.Div(
                [
                    html.H4("Model Status", style={"margin": "0", "color": "#2c3e50"}),
                    html.P(
                        status_text,
                        style={
                            "margin": "5px 0",
                            "color": status_color,
                            "fontWeight": "bold",
                        },
                    ),
                    html.P(
                        f"Version: {model_metadata.get('version', 'Unknown')}",
                        style={"margin": "5px 0", "color": "#7f8c8d"},
                    ),
                    html.P(
                        f"Stage: {model_metadata.get('stage', 'Unknown')}",
                        style={"margin": "5px 0", "color": "#7f8c8d"},
                    ),
                ],
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "width": "300px",
                    "display": "inline-block",
                    "margin": "10px",
                },
            ),
            # Prediction Stats Card (placeholder)
            html.Div(
                [
                    html.H4(
                        "Prediction Stats", style={"margin": "0", "color": "#2c3e50"}
                    ),
                    html.P(
                        "Total Predictions: 0",
                        style={"margin": "5px 0", "color": "#7f8c8d"},
                    ),
                    html.P(
                        "Last Prediction: None",
                        style={"margin": "5px 0", "color": "#7f8c8d"},
                    ),
                ],
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "width": "300px",
                    "display": "inline-block",
                    "margin": "10px",
                },
            ),
        ]
    )


def create_manual_input() -> html.Div:
    """Create manual input form with actual personality features.

    Returns:
        Div containing manual input components
    """
    return html.Div(
        [
            html.P(
                "Enter your personality traits below:",
                style={"fontSize": "16px", "marginBottom": "20px", "color": "#2c3e50"},
            ),

            # Time spent alone
            html.Div(
                [
                    html.Label(
                        "Time Spent Alone (hours per day):",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Input(
                        id="time-spent-alone",
                        type="number",
                        value=2.0,
                        min=0,
                        max=24,
                        step=0.5,
                        style={"margin": "5px", "width": "200px", "padding": "5px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Social event attendance
            html.Div(
                [
                    html.Label(
                        "Social Event Attendance (events per month):",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Input(
                        id="social-event-attendance",
                        type="number",
                        value=4.0,
                        min=0,
                        max=30,
                        step=1,
                        style={"margin": "5px", "width": "200px", "padding": "5px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Going outside
            html.Div(
                [
                    html.Label(
                        "Going Outside (frequency per week):",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Input(
                        id="going-outside",
                        type="number",
                        value=3.0,
                        min=0,
                        max=7,
                        step=1,
                        style={"margin": "5px", "width": "200px", "padding": "5px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Friends circle size
            html.Div(
                [
                    html.Label(
                        "Friends Circle Size:",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Input(
                        id="friends-circle-size",
                        type="number",
                        value=8.0,
                        min=0,
                        max=50,
                        step=1,
                        style={"margin": "5px", "width": "200px", "padding": "5px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Post frequency
            html.Div(
                [
                    html.Label(
                        "Social Media Post Frequency (posts per week):",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Input(
                        id="post-frequency",
                        type="number",
                        value=3.0,
                        min=0,
                        max=20,
                        step=1,
                        style={"margin": "5px", "width": "200px", "padding": "5px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Stage fear
            html.Div(
                [
                    html.Label(
                        "Do you have stage fear?",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Dropdown(
                        id="stage-fear",
                        options=[
                            {"label": "No", "value": "No"},
                            {"label": "Yes", "value": "Yes"},
                            {"label": "Unknown", "value": "Unknown"},
                        ],
                        value="No",
                        style={"width": "200px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),

            # Drained after socializing
            html.Div(
                [
                    html.Label(
                        "Do you feel drained after socializing?",
                        style={"display": "block", "fontWeight": "bold", "marginBottom": "5px"},
                    ),
                    dcc.Dropdown(
                        id="drained-after-socializing",
                        options=[
                            {"label": "No", "value": "No"},
                            {"label": "Yes", "value": "Yes"},
                            {"label": "Unknown", "value": "Unknown"},
                        ],
                        value="No",
                        style={"width": "200px"},
                    ),
                ],
                style={"marginBottom": "15px"},
            ),
        ],
        id="manual-inputs",
        style={
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "10px",
            "border": "1px solid #dee2e6",
        },
    )


def format_prediction_result(result: dict[str, Any]) -> html.Div:
    """Format prediction result for display.

    Args:
        result: Prediction result dictionary

    Returns:
        Formatted result component
    """
    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0)
    prob_extrovert = result.get("probability_extrovert", 0)
    prob_introvert = result.get("probability_introvert", 0)

    # Create visual elements
    confidence_color = (
        "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
    )

    # Choose personality color
    personality_color = "#e74c3c" if prediction == "Extrovert" else "#3498db"

    elements = [
        html.H4(
            "Personality Classification Result",
            style={"color": "#2c3e50", "marginBottom": "15px"}
        ),
        # Main prediction with personality-specific styling
        html.Div(
            [
                html.H2(
                    f"🧠 You are classified as: {prediction}",
                    style={
                        "color": personality_color,
                        "margin": "10px 0",
                        "textAlign": "center",
                        "backgroundColor": "#ecf0f1",
                        "padding": "15px",
                        "borderRadius": "10px",
                        "border": f"2px solid {personality_color}"
                    },
                )
            ]
        ),
        # Confidence score
        html.Div(
            [
                html.P(
                    f"Confidence Score: {confidence:.1%}",
                    style={
                        "fontSize": "18px",
                        "color": confidence_color,
                        "margin": "15px 0",
                        "textAlign": "center",
                        "fontWeight": "bold",
                    },
                )
            ]
        ),
    ]

    # Add detailed probability breakdown
    if prob_extrovert is not None and prob_introvert is not None:
        elements.append(
            html.Div(
                [
                    html.H5("Detailed Probabilities:", style={"margin": "20px 0 10px 0", "color": "#2c3e50"}),
                    html.Div(
                        [
                            # Extrovert bar
                            html.Div(
                                [
                                    html.Span("Extrovert: ", style={"fontWeight": "bold", "width": "100px", "display": "inline-block"}),
                                    html.Div(
                                        style={
                                            "backgroundColor": "#e74c3c",
                                            "width": f"{prob_extrovert * 100}%",
                                            "height": "20px",
                                            "borderRadius": "10px",
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "minWidth": "2px",
                                        }
                                    ),
                                    html.Span(f"{prob_extrovert:.1%}", style={"fontWeight": "bold"}),
                                ],
                                style={"margin": "10px 0", "display": "flex", "alignItems": "center"},
                            ),
                            # Introvert bar
                            html.Div(
                                [
                                    html.Span("Introvert: ", style={"fontWeight": "bold", "width": "100px", "display": "inline-block"}),
                                    html.Div(
                                        style={
                                            "backgroundColor": "#3498db",
                                            "width": f"{prob_introvert * 100}%",
                                            "height": "20px",
                                            "borderRadius": "10px",
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "minWidth": "2px",
                                        }
                                    ),
                                    html.Span(f"{prob_introvert:.1%}", style={"fontWeight": "bold"}),
                                ],
                                style={"margin": "10px 0", "display": "flex", "alignItems": "center"},
                            ),
                        ],
                        style={
                            "backgroundColor": "#f8f9fa",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "border": "1px solid #dee2e6",
                        },
                    ),
                ]
            )
        )

    # Add personality description
    if prediction == "Extrovert":
        description = "🎉 Extroverts typically enjoy social situations, feel energized by being around people, and tend to be outgoing and expressive."
        description_color = "#e74c3c"
    elif prediction == "Introvert":
        description = "🤔 Introverts typically prefer quieter environments, feel energized by alone time, and tend to be more reflective and reserved."
        description_color = "#3498db"
    else:
        description = "The model could not clearly determine your personality type."
        description_color = "#7f8c8d"

    elements.append(
        html.Div(
            [
                html.P(
                    description,
                    style={
                        "fontSize": "14px",
                        "color": description_color,
                        "margin": "15px 0",
                        "padding": "10px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "5px",
                        "fontStyle": "italic",
                    },
                )
            ]
        )
    )

    # Add metadata
    elements.append(
        html.Div(
            [
                html.Hr(style={"margin": "20px 0"}),
                html.P(
                    f"Model: {result.get('model_name', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0", "fontSize": "12px"},
                ),
                html.P(
                    f"Version: {result.get('model_version', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0", "fontSize": "12px"},
                ),
                html.P(
                    f"Timestamp: {result.get('timestamp', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0", "fontSize": "12px"},
                ),
            ]
        )
    )

    return html.Div(
        elements,
        style={
            "border": "2px solid " + confidence_color,
            "padding": "20px",
            "borderRadius": "10px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )
