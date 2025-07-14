"""Model loading and management for the Dash application."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelLoader:
    """Handles loading and managing ML models from various sources."""

    def __init__(
        self,
        model_name: str,
        model_version: str | None = None,
        model_stage: str = "Production",
    ):
        """Initialize the model loader.

        Args:
            model_name: Name of the model to load
            model_version: Specific version to load (optional)
            model_stage: Stage to load from if version not specified
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_stage = model_stage
        self.logger = logging.getLogger(__name__)

        self.model: Any = None
        self.model_metadata: dict[str, Any] = {}

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the model or create a dummy model for demonstration."""
        try:
            # Try to load from local files first
            model_dir = Path("best_params")
            if model_dir.exists():
                # Look for saved models
                for model_file in model_dir.glob("*.pkl"):
                    try:
                        with open(model_file, "rb") as f:
                            self.model = pickle.load(f)  # nosec B301
                        self.logger.info(f"Loaded model from {model_file}")
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_file}: {e}")
                        continue

            # If no model found, create dummy model
            if self.model is None:
                self._create_dummy_model()

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # For demo purposes, create a dummy model
            self._create_dummy_model()

    def _create_dummy_model(self) -> None:
        """Create a dummy model for demonstration purposes."""

        class DummyModel:
            def predict(self, X):
                # Simple dummy prediction for 6 personality types
                return np.random.choice([0, 1, 2, 3, 4, 5], size=len(X))

            def predict_proba(self, X):
                # Simple dummy probabilities for 6 classes
                probs = np.random.random((len(X), 6))
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs

        self.model = DummyModel()
        self.model_metadata = {
            "version": "dummy",
            "stage": "Development",
            "description": "Dummy personality classification model for demonstration",
        }
        self.logger.info("Created dummy personality model for demonstration")

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single instance.

        Args:
            data: Input data for prediction

        Returns:
            Prediction result
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Convert input to DataFrame
            if "features" in data:
                features = pd.DataFrame([data["features"]])
            else:
                features = pd.DataFrame([data])

            # Make prediction
            prediction = self.model.predict(features)[0]

            # Get prediction probabilities if available
            probabilities = None
            confidence = None

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                probabilities = proba.tolist()
                confidence = float(max(proba))

            result = {
                "prediction": prediction.tolist()
                if hasattr(prediction, "tolist")
                else prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "model_name": self.model_name,
                "model_version": self.model_metadata.get("version"),
            }

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Model metadata dictionary
        """
        return self.model_metadata.copy()

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None


class LocalModelLoader:
    """Loads models from local files."""

    def __init__(self, model_path: str | Path, model_name: str = "personality_model"):
        """Initialize local model loader.

        Args:
            model_path: Path to the saved model
            model_name: Name of the model
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load model from disk."""
        try:
            if self.model_path.suffix == ".pkl":
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)  # nosec B301
            elif self.model_path.suffix == ".joblib":
                self.model = joblib.load(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

            self.logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Make predictions.

        Args:
            features: Input features

        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Convert to DataFrame if needed
            if isinstance(features, dict):
                features = pd.DataFrame([features])

            # Make prediction
            predictions = self.model.predict(features)

            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)

            result = {
                "predictions": predictions.tolist()
                if hasattr(predictions, "tolist")
                else predictions,
                "probabilities": probabilities.tolist()
                if probabilities is not None
                else None,
                "model_name": self.model_name,
            }

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
