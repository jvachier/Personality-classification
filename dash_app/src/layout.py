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
                    # Input methods tabs
                    dcc.Tabs(
                        id="input-tabs",
                        value="manual",
                        children=[
                            dcc.Tab(label="Manual Input", value="manual"),
                            dcc.Tab(label="JSON Input", value="json"),
                            dcc.Tab(label="File Upload", value="file"),
                        ],
                    ),
                    # Input content
                    html.Div(
                        id="input-content",
                        style={"marginTop": "20px"},
                        children=[create_manual_input()],
                    ),  # Start with manual input
                    # Hidden placeholder components for callback states
                    html.Div(
                        [
                            dcc.Textarea(id="json-input", style={"display": "none"}),
                            dcc.Upload(id="file-upload", style={"display": "none"}),
                        ],
                        style={"display": "none"},
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
    """Create manual input form.

    Returns:
        Div containing manual input components
    """
    return html.Div(
        [
            html.P("Manual feature input (demo - replace with actual feature inputs):"),
            html.Div(
                [
                    html.Label(
                        "Feature 1:",
                        style={"display": "inline-block", "width": "100px"},
                    ),
                    dcc.Input(
                        id="feature1",
                        type="number",
                        value=0.5,
                        style={"margin": "5px", "width": "100px"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label(
                        "Feature 2:",
                        style={"display": "inline-block", "width": "100px"},
                    ),
                    dcc.Input(
                        id="feature2",
                        type="number",
                        value=0.3,
                        style={"margin": "5px", "width": "100px"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label(
                        "Feature 3:",
                        style={"display": "inline-block", "width": "100px"},
                    ),
                    dcc.Input(
                        id="feature3",
                        type="number",
                        value=0.8,
                        style={"margin": "5px", "width": "100px"},
                    ),
                ]
            ),
        ],
        id="manual-inputs",
    )


def create_json_input() -> html.Div:
    """Create JSON input form.

    Returns:
        Div containing JSON input components
    """
    example_json = """{\n  "feature1": 0.5,\n  "feature2": 0.3,\n  "feature3": 0.8\n}"""
    return html.Div(
        [
            html.P("Enter JSON data for prediction:"),
            dcc.Textarea(
                id="json-input-display",  # Different ID to avoid conflicts
                placeholder=example_json,
                value=example_json,
                style={"width": "100%", "height": 150, "fontFamily": "monospace"},
            ),
        ]
    )


def create_file_input() -> html.Div:
    """Create file upload form.

    Returns:
        Div containing file upload components
    """
    return html.Div(
        [
            html.P("Upload CSV file for batch predictions:"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
            ),
        ]
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
    probabilities = result.get("probabilities", [])

    # Create visual elements
    confidence_color = (
        "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
    )

    elements = [
        html.H4(
            "Prediction Result", style={"color": "#2c3e50", "marginBottom": "15px"}
        ),
        # Main prediction
        html.Div(
            [
                html.H2(
                    f"Prediction: {prediction}",
                    style={"color": confidence_color, "margin": "10px 0"},
                )
            ]
        ),
        # Confidence score
        html.Div(
            [
                html.P(
                    f"Confidence: {confidence:.3f}",
                    style={
                        "fontSize": "18px",
                        "color": confidence_color,
                        "margin": "10px 0",
                    },
                )
            ]
        ),
    ]

    # Add probability breakdown if available
    if probabilities:
        elements.append(
            html.Div(
                [
                    html.H5("Class Probabilities:", style={"margin": "15px 0 10px 0"}),
                    html.Ul(
                        [
                            html.Li(f"Class {i}: {prob:.3f}", style={"margin": "5px 0"})
                            for i, prob in enumerate(probabilities)
                        ]
                    ),
                ]
            )
        )

    # Add metadata
    elements.append(
        html.Div(
            [
                html.Hr(),
                html.P(
                    f"Model: {result.get('model_name', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0"},
                ),
                html.P(
                    f"Version: {result.get('model_version', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0"},
                ),
                html.P(
                    f"Timestamp: {result.get('timestamp', 'Unknown')}",
                    style={"color": "#7f8c8d", "margin": "5px 0"},
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
            "backgroundColor": "#f8f9fa",
        },
    )
