"""Main entry point for the Dash application."""

import argparse
import logging

from src import PersonalityClassifierApp


def main():
    """Main function to run the Dash application."""
    parser = argparse.ArgumentParser(description="Personality Classification Dash App")
    parser.add_argument(
        "--model-name", required=True, help="Name of the model to serve"
    )
    parser.add_argument("--model-version", help="Specific version to serve")
    parser.add_argument(
        "--model-stage", default="Production", help="Stage to serve from"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create and run Dash app
    app = PersonalityClassifierApp(
        model_name=args.model_name,
        model_version=args.model_version,
        model_stage=args.model_stage,
        host=args.host,
        port=args.port,
    )

    print(f"Starting Dash Model Server for '{args.model_name}'")
    print(f"Dashboard available at: http://{args.host}:{args.port}")
    print("Use the interactive dashboard to make predictions!")

    app.run(debug=args.debug)


if __name__ == "__main__":
    main()
