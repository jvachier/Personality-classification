"""Layout components for the Dash application."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html


def create_layout(model_name: str, model_metadata: dict[str, Any]) -> html.Div:
    """Create the main layout for the Dash application.

    Args:
        model_name: Name of the model
        model_metadata: Model metadata dictionary

    Returns:
        Dash HTML layout
    """
    return html.Div([
        # Professional Header
        create_professional_header(),

        # Main Content
        dbc.Container([
            dbc.Row([
                # Input Panel - Original size
                dbc.Col([
                    create_input_panel()
                ], md=5, className="d-flex align-self-stretch"),

                # Results Panel - Original size
                dbc.Col([
                    html.Div(id="prediction-results", children=[
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Analysis Results", className="mb-0 text-center",
                                       style={"color": "#2c3e50", "fontWeight": "400"})
                            ], style={"backgroundColor": "#ffffff", "border": "none"}),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-chart-radar fa-3x mb-3",
                                          style={"color": "#bdc3c7"}),
                                    html.H5("Ready for Analysis",
                                           style={"color": "#7f8c8d"}),
                                    html.P("Adjust the parameters and click 'Analyze Personality' to see your results.",
                                          style={"color": "#95a5a6"})
                                ], className="text-center py-5")
                            ], style={"padding": "2rem"})
                        ], className="shadow-sm h-100",
                          style={"border": "none", "borderRadius": "15px"})
                    ], className="h-100 d-flex flex-column")
                ], md=7, className="d-flex align-self-stretch")
            ], justify="center", className="g-4", style={"minHeight": "80vh"})
        ], fluid=True, className="py-4", style={"backgroundColor": "#ffffff", "maxWidth": "1400px", "margin": "0 auto"})
    ], className="personality-dashboard", style={
        "backgroundColor": "#ffffff !important",
        "minHeight": "100vh"
    })


def create_professional_header() -> dbc.Row:
    """Create a professional header."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-brain me-3", style={"fontSize": "2.5rem", "color": "#2c3e50"}),
                            html.H1("Personality Classification", className="d-inline-block mb-0",
                                   style={"color": "#2c3e50", "fontWeight": "300"}),
                        ], className="d-flex align-items-center justify-content-center"),

                        html.P(
                            "Advanced AI-powered personality assessment platform using ensemble machine learning to analyze behavioral patterns and predict introversion-extraversion tendencies based on social, lifestyle, and digital behavior indicators.",
                            className="text-center text-muted mt-2 mb-0",
                            style={"fontSize": "1.0rem", "maxWidth": "800px", "margin": "0 auto"}
                        )
                    ], className="py-3")
                ], className="shadow-sm border-0", style={"backgroundColor": "#ffffff"})
            ])
        ], className="mb-4")
    ], fluid=True, style={"maxWidth": "1400px", "margin": "0 auto"})


def create_input_panel() -> dbc.Card:
    """Create a clean, professional input panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Assessment Parameters", className="mb-0 text-center",
                   style={"color": "#2c3e50", "fontWeight": "400"})
        ], style={"backgroundColor": "#ffffff", "border": "none"}),
        dbc.CardBody([
            # Social Behavior Section
            html.H5([
                html.I(className="fas fa-users me-2", style={"color": "#3498db"}),
                "Social Behavior"
            ], className="section-title mb-4"),

            create_enhanced_slider(
                "time-spent-alone",
                "Time Spent Alone (hours/day)",
                0, 24, 8,
                "Less alone time",
                "More alone time",
                "slider-social"
            ),

            create_enhanced_slider(
                "social-event-attendance",
                "Social Event Attendance (events/month)",
                0, 20, 4,
                "Fewer events",
                "More events",
                "slider-social"
            ),

            # Lifestyle Section
            html.H5([
                html.I(className="fas fa-compass me-2", style={"color": "#27ae60"}),
                "Lifestyle"
            ], className="section-title mt-5 mb-4"),

            create_enhanced_slider(
                "going-outside",
                "Going Outside Frequency (times/week)",
                0, 15, 5,
                "Stay indoors",
                "Go out frequently",
                "slider-lifestyle"
            ),

            create_enhanced_slider(
                "friends-circle-size",
                "Friends Circle Size",
                0, 50, 12,
                "Small circle",
                "Large network",
                "slider-lifestyle"
            ),

            # Digital Behavior Section
            html.H5([
                html.I(className="fas fa-share-alt me-2", style={"color": "#9b59b6"}),
                "Digital Behavior"
            ], className="section-title mt-5 mb-4"),

            create_enhanced_slider(
                "post-frequency",
                "Social Media Posts (per week)",
                0, 20, 3,
                "Rarely post",
                "Frequently post",
                "slider-digital"
            ),

            # Psychological Assessment Section
            html.H5([
                html.I(className="fas fa-mind-share me-2", style={"color": "#e67e22"}),
                "Psychological Assessment"
            ], className="section-title mt-5 mb-4"),

            create_enhanced_dropdown(
                "stage-fear",
                "Do you have stage fear?",
                [
                    {"label": "No - I'm comfortable with public speaking", "value": "No"},
                    {"label": "Yes - I avoid speaking in public", "value": "Yes"},
                    {"label": "Sometimes - It depends on the situation", "value": "Unknown"}
                ],
                "No"
            ),

            create_enhanced_dropdown(
                "drained-after-socializing",
                "Do you feel drained after socializing?",
                [
                    {"label": "No - I feel energized by social interaction", "value": "No"},
                    {"label": "Yes - I need time alone to recharge", "value": "Yes"},
                    {"label": "It varies - Depends on the context", "value": "Unknown"}
                ],
                "No"
            ),

            # Analysis Button
            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-brain me-2"),
                        "Analyze Personality"
                    ],
                    id="predict-button",
                    color="primary",
                    size="lg",
                    className="predict-button px-5 py-3",
                    style={"fontSize": "1.1rem", "fontWeight": "500"}
                )
            ], className="text-center mt-5")
        ], style={"padding": "2rem"})
    ], className="shadow-sm h-100", style={"border": "none", "borderRadius": "15px"})


def create_enhanced_slider(slider_id: str, label: str, min_val: int, max_val: int,
                          default: int, intro_text: str, extro_text: str,
                          css_class: str) -> html.Div:
    """Create an enhanced slider with personality hints."""
    return html.Div([
        html.Label(label, className="slider-label fw-bold"),
        dcc.Slider(
            id=slider_id,
            min=min_val,
            max=max_val,
            step=1,
            value=default,
            marks={
                min_val: {"label": intro_text, "style": {"color": "#3498db", "fontSize": "0.8rem"}},
                max_val: {"label": extro_text, "style": {"color": "#e74c3c", "fontSize": "0.8rem"}}
            },
            tooltip={"placement": "bottom", "always_visible": True},
            className=f"personality-slider {css_class}"
        )
    ], className="slider-container mb-3")


def create_enhanced_dropdown(dropdown_id: str, label: str, options: list,
                           default: str) -> html.Div:
    """Create an enhanced dropdown with better styling."""
    return html.Div([
        html.Label(label, className="dropdown-label fw-bold"),
        dcc.Dropdown(
            id=dropdown_id,
            options=options,
            value=default,
            className="personality-dropdown"
        )
    ], className="dropdown-container mb-3")


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
    input_data = result.get("input_data", {})

    # Determine confidence level
    if confidence > 0.7:
        confidence_color = "success"
        confidence_badge = "High Confidence"
    elif confidence > 0.5:
        confidence_color = "warning"
        confidence_badge = "Medium Confidence"
    else:
        confidence_color = "danger"
        confidence_badge = "Low Confidence"

    # Create enhanced results with Bootstrap components
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Analysis Results", className="mb-0 text-center",
                   style={"color": "#2c3e50", "fontWeight": "400"})
        ], style={"backgroundColor": "#ffffff", "border": "none"}),
        dbc.CardBody([
            dbc.Row([
                # Main Result
                dbc.Col([
                    html.Div([
                        html.H2(
                            f"🧠 {prediction}",
                            className="personality-result text-center"
                        ),
                        html.P(
                            f"Confidence: {confidence:.1%}",
                            className="confidence-score text-center"
                        ),
                        dbc.Badge(
                            confidence_badge,
                            color=confidence_color,
                            className="mb-3"
                        )
                    ], className="text-center")
                ], md=6),

                # Confidence Bars
                dbc.Col([
                    html.H5("Probability Breakdown"),
                    create_confidence_bars({
                        "Extrovert": prob_extrovert,
                        "Introvert": prob_introvert
                    })
                ], md=6)
            ]),

            # Larger Radar Chart - Full Width
            dbc.Row([
                dbc.Col([
                    html.H5("Personality Dimensions", className="text-center mb-3"),
                    html.Div([
                        dcc.Graph(
                            figure=create_personality_radar({
                                "Introvert": prob_introvert,
                                "Extrovert": prob_extrovert
                            }, input_data),
                            config={"displayModeBar": False},
                            className="personality-radar",
                            style={"height": "450px", "width": "100%"}
                        )
                    ], style={"padding": "0 20px"})
                ], md=12, className="text-center")
            ], className="mt-4"),

            # Personality Insights
            html.Hr(),
            html.Div([
                html.H5("Personality Insights"),
                create_personality_insights(prediction, confidence)
            ]),

            # Metadata
            html.Hr(),
            html.Small([
                f"Model: {result.get('model_name', 'Unknown')} | ",
                f"Version: {result.get('model_version', 'Unknown')} | ",
                f"Timestamp: {result.get('timestamp', 'Unknown')}"
            ], className="text-muted")
        ], style={"padding": "2rem"})
    ], className="shadow-sm h-100", style={"border": "none", "borderRadius": "15px"})


def create_confidence_bars(probabilities: dict) -> html.Div:
    """Create animated confidence bars."""
    bars = []
    for personality, prob in probabilities.items():
        color = "primary" if personality == "Introvert" else "danger"
        bars.append(
            html.Div([
                html.Span(personality, className="personality-label"),
                dbc.Progress(
                    value=prob * 100,
                    color=color,
                    className="confidence-bar mb-2",
                    animated=True,
                    striped=True
                ),
                html.Span(f"{prob:.1%}", className="confidence-text")
            ], className="confidence-row mb-2")
        )
    return html.Div(bars)


def create_personality_insights(prediction: str, confidence: float) -> html.Div:
    """Create personality insights based on prediction."""
    insights = {
        "Introvert": [
            "� You likely process information internally before sharing",
            "⚡ You recharge through quiet, solitary activities",
            "👥 You prefer deep, meaningful conversations over small talk",
            "🎯 You tend to think before speaking"
        ],
        "Extrovert": [
            "🗣️ You likely think out loud and enjoy verbal processing",
            "⚡ You gain energy from social interactions",
            "👥 You enjoy meeting new people and large gatherings",
            "🎯 You tend to speak spontaneously"
        ]
    }

    prediction_insights = insights.get(prediction, ["Analysis in progress..."])

    return html.Ul([
        html.Li(insight, className="insight-item")
        for insight in prediction_insights
    ], className="insights-list")


def create_personality_radar(probabilities: dict, input_data: dict[str, Any] | None = None) -> go.Figure:
    """Create radar chart for personality visualization."""
    categories = ["Social Energy", "Processing Style", "Decision Making",
                 "Lifestyle", "Communication"]

    # Calculate values based on probabilities and input data
    intro_tendency = probabilities.get("Introvert", 0.5)

    # Map input data to personality dimensions (simplified)
    if input_data:
        social_energy = 1 - (input_data.get("Time_spent_Alone", 12) / 24)
        processing_style = 1 - (input_data.get("Post_frequency", 10) / 20)
        decision_making = 0.8 if input_data.get("Stage_fear_Yes", 0) else 0.3
        lifestyle = 1 - (input_data.get("Going_outside", 7) / 15)
        communication = 1 - (input_data.get("Friends_circle_size", 25) / 50)

        values = [social_energy, processing_style, decision_making, lifestyle, communication]
    else:
        # Default values based on prediction
        values = [intro_tendency] * len(categories)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='#3498db' if intro_tendency > 0.5 else '#e74c3c'
    ))

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1]},
            "angularaxis": {"tickfont": {"size": 12}}
        },
        showlegend=False,
        height=450,
        font={"size": 12},
        title="Personality Dimensions",
        margin={"l": 80, "r": 80, "t": 60, "b": 80}
    )

    return fig
