"""Enhanced layout with modern UI/UX improvements."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html


def create_enhanced_layout(model_name: str, model_metadata: dict[str, Any]) -> html.Div:
    """Create an enhanced layout with modern UI/UX improvements.

    Args:
        model_name: Name of the model
        model_metadata: Model metadata dictionary

    Returns:
        Enhanced Dash HTML layout
    """
    return dbc.Container(
        [
            # Enhanced Header
            create_enhanced_header(),

            # Main Content Grid
            dbc.Row([
                # Input Panel
                dbc.Col([
                    create_input_panel()
                ], md=8),

                # Live Feedback Panel
                dbc.Col([
                    create_feedback_panel()
                ], md=4)
            ], className="mb-4"),

            # Results Section
            dbc.Row([
                dbc.Col([
                    html.Div(id="enhanced-results")
                ])
            ])
        ],
        fluid=True,
        className="personality-dashboard"
    )


def create_enhanced_header() -> dbc.Row:
    """Create an enhanced header with branding."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fas fa-brain me-3", style={"fontSize": "2rem", "color": "#3498db"}),
                html.H1("PersonalityAI", className="d-inline-block mb-0"),
                html.Span(" Classification Dashboard", className="text-muted ms-2")
            ], className="d-flex align-items-center justify-content-center"),

            # Breadcrumb navigation
            dbc.Breadcrumb([
                {"label": "Home", "href": "#", "external_link": True},
                {"label": "Dashboard", "active": True}
            ], className="justify-content-center mt-2")
        ])
    ], className="text-center mb-4")


def create_input_panel() -> dbc.Card:
    """Create enhanced input panel with modern controls."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-sliders-h me-2"),
            html.H4("Personality Assessment", className="mb-0")
        ]),
        dbc.CardBody([
            # Social Behavior Section
            html.H5([
                html.I(className="fas fa-users me-2", style={"color": "#e74c3c"}),
                "Social Behavior"
            ], className="section-title"),

            create_enhanced_slider(
                "time-spent-alone",
                "Time Spent Alone (hours/day)",
                0, 24, 2,
                "🏠 Recharge in solitude",
                "👥 Energy from others",
                "slider-social"
            ),

            create_enhanced_slider(
                "social-event-attendance",
                "Social Event Attendance (events/month)",
                0, 20, 4,
                "🏡 Prefer staying in",
                "🎉 Love social gatherings",
                "slider-social"
            ),

            # Lifestyle Section
            html.H5([
                html.I(className="fas fa-home me-2", style={"color": "#27ae60"}),
                "Lifestyle Preferences"
            ], className="section-title mt-4"),

            create_enhanced_slider(
                "going-outside",
                "Going Outside Frequency (times/week)",
                0, 15, 3,
                "🏠 Homebody",
                "🌍 Adventure seeker",
                "slider-lifestyle"
            ),

            create_enhanced_slider(
                "friends-circle-size",
                "Friends Circle Size",
                0, 50, 8,
                "👤 Few close friends",
                "👥 Large social network",
                "slider-lifestyle"
            ),

            # Digital Behavior Section
            html.H5([
                html.I(className="fas fa-mobile-alt me-2", style={"color": "#9b59b6"}),
                "Digital Behavior"
            ], className="section-title mt-4"),

            create_enhanced_slider(
                "post-frequency",
                "Social Media Posts (per week)",
                0, 20, 3,
                "📱 Lurker",
                "📢 Active sharer",
                "slider-digital"
            ),

            # Psychological Traits Section
            html.H5([
                html.I(className="fas fa-brain me-2", style={"color": "#f39c12"}),
                "Psychological Traits"
            ], className="section-title mt-4"),

            create_enhanced_dropdown(
                "stage-fear",
                "Do you have stage fear?",
                [
                    {"label": "🚫 No - I enjoy being center of attention", "value": "No"},
                    {"label": "😰 Yes - I avoid public speaking", "value": "Yes"},
                    {"label": "🤔 Unknown/Sometimes", "value": "Unknown"}
                ],
                "No"
            ),

            create_enhanced_dropdown(
                "drained-after-socializing",
                "Do you feel drained after socializing?",
                [
                    {"label": "⚡ No - I feel energized", "value": "No"},
                    {"label": "😴 Yes - I need alone time", "value": "Yes"},
                    {"label": "🤷 Depends on the situation", "value": "Unknown"}
                ],
                "No"
            ),

            # Prediction Button
            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-magic me-2"),
                        "Analyze My Personality"
                    ],
                    id="predict-button",
                    color="primary",
                    size="lg",
                    className="predict-button"
                )
            ], className="text-center mt-4")
        ])
    ], className="input-panel")


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
        ),
        html.Small(
            f"Current value reflects your tendency on the introversion-extraversion spectrum",
            className="text-muted slider-help"
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


def create_feedback_panel() -> dbc.Card:
    """Create real-time feedback panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-line me-2"),
            html.H5("Live Assessment", className="mb-0")
        ]),
        dbc.CardBody([
            # Personality Meter
            html.Div([
                html.H6("Current Tendency", className="text-center"),
                html.Div(id="personality-meter", className="meter-container"),
                html.Div([
                    html.Span("Introvert", className="meter-label intro"),
                    html.Span("Extrovert", className="meter-label extro")
                ], className="d-flex justify-content-between mt-2")
            ], className="mb-4"),

            # Quick Insights
            html.Div([
                html.H6("Quick Insights"),
                html.Div(id="live-insights", className="insights-container")
            ])
        ])
    ], className="feedback-panel")


def create_enhanced_results(result: dict[str, Any]) -> html.Div:
    """Create enhanced results display with visualizations."""
    if not result:
        return html.Div()

    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', {})

    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-user-circle me-2"),
            html.H4("Your Personality Analysis", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                # Main Result
                dbc.Col([
                    html.Div([
                        html.H2(prediction, className="personality-result"),
                        html.P(f"Confidence: {confidence:.1%}", className="confidence-score"),
                        create_confidence_bars(probabilities)
                    ], className="text-center")
                ], md=6),

                # Radar Chart
                dbc.Col([
                    dcc.Graph(
                        figure=create_personality_radar(probabilities),
                        config={"displayModeBar": False},
                        className="personality-radar"
                    )
                ], md=6)
            ]),

            # Personality Insights
            html.Hr(),
            html.Div([
                html.H5("Personality Insights"),
                create_personality_insights(prediction, confidence)
            ])
        ])
    ], className="results-panel mt-4")


def create_confidence_bars(probabilities: dict) -> html.Div:
    """Create animated confidence bars."""
    bars = []
    for personality, prob in probabilities.items():
        color = "#3498db" if personality == "Introvert" else "#e74c3c"
        bars.append(
            html.Div([
                html.Span(personality, className="personality-label"),
                dbc.Progress(
                    value=prob * 100,
                    color="primary" if personality == "Introvert" else "danger",
                    className="confidence-bar",
                    animated=True,
                    striped=True
                ),
                html.Span(f"{prob:.1%}", className="confidence-text")
            ], className="confidence-row mb-2")
        )
    return html.Div(bars)


def create_personality_radar(probabilities: dict) -> go.Figure:
    """Create radar chart for personality visualization."""
    categories = ["Social Energy", "Processing Style", "Decision Making",
                 "Lifestyle", "Communication"]

    # Mock values based on probabilities (in real implementation,
    # you'd extract these from detailed model outputs)
    values = [probabilities.get("Introvert", 0.5)] * len(categories)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='#3498db'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=300
    )

    return fig


def create_personality_insights(prediction: str, confidence: float) -> html.Div:
    """Create personality insights based on prediction."""
    insights = {
        "Introvert": [
            "🧠 You likely process information internally before sharing",
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
