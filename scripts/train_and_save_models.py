#!/usr/bin/env python3
"""Train and save personality classification models for Dash app serving."""

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from src.modules.config import RND, setup_logging
from src.modules.data_loader import load_data_with_external_merge
from src.modules.model_builders import (
    build_neural_stack,
    build_noisy_stack,
    build_sklearn_stack,
    build_stack,
    build_stack_c,
)
from src.modules.preprocessing import prep
from src.modules.utils import get_logger


def load_best_params(stack_name: str) -> dict[str, Any]:
    """Load best parameters for a stack from JSON file."""
    params_file = Path("best_params") / f"stack_{stack_name}_best_params.json"

    if not params_file.exists():
        raise FileNotFoundError(f"Best parameters file not found: {params_file}")

    with open(params_file) as f:
        params = json.load(f)

    return params


def create_mock_trial(params: dict[str, Any]) -> Any:
    """Create a mock trial object with suggest methods that return the saved parameters."""

    class MockTrial:
        def __init__(self, params: dict[str, Any]):
            self.params = params

        def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
            return float(self.params[name])

        def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
            return int(self.params[name])

        def suggest_categorical(self, name: str, choices: list, **kwargs) -> Any:
            return self.params[name]

    return MockTrial(params)


def train_and_save_stack(stack_name: str, data_dict: dict, save_dir: Path) -> None:
    """Train and save a single stack model."""
    logger = get_logger(__name__)
    logger.info(f"Training and saving stack {stack_name}...")

    # Load best parameters
    try:
        best_params = load_best_params(stack_name)
        mock_trial = create_mock_trial(best_params)
    except FileNotFoundError:
        logger.warning(f"No best parameters found for stack {stack_name}, skipping...")
        return

    # Build model based on stack type
    if stack_name == "A":
        model = build_stack(mock_trial, seed=RND, wide_hp=False)
    elif stack_name == "B":
        model = build_stack(mock_trial, seed=2024, wide_hp=True)
    elif stack_name == "C":
        model = build_stack_c(mock_trial, seed=1337)
    elif stack_name == "D":
        model = build_sklearn_stack(mock_trial, seed=9999, X_full=data_dict["X_full"])
    elif stack_name == "E":
        model = build_neural_stack(mock_trial, seed=7777, X_full=data_dict["X_full"])
    elif stack_name == "F":
        model = build_noisy_stack(mock_trial, seed=5555, noise_rate=0.1)
    else:
        raise ValueError(f"Unknown stack: {stack_name}")

    # Train model on full training data
    X_train = data_dict["X_full"]
    y_train = data_dict["y_full"]

    logger.info(f"Training {stack_name} on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Save model
    model_file = save_dir / f"stack_{stack_name}_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Saved stack {stack_name} to {model_file}")

    # Save metadata
    metadata = {
        "stack_name": stack_name,
        "model_type": type(model).__name__,
        "best_params": best_params,
        "training_samples": len(X_train),
        "features": X_train.columns.tolist(),
        "target_classes": sorted(y_train.unique().tolist()),
    }

    metadata_file = save_dir / f"stack_{stack_name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata for stack {stack_name} to {metadata_file}")


def train_ensemble_model(data_dict: dict, save_dir: Path) -> None:
    """Train and save a simple ensemble model."""
    logger = get_logger(__name__)
    logger.info("Training ensemble model...")

    # For now, we'll create a simple voting ensemble
    # In practice, this would use the blending weights from the main pipeline

    # Create a simple ensemble with different algorithms
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=RND, max_iter=1000)),
            ("rf", RandomForestClassifier(random_state=RND, n_estimators=100)),
            ("lgb", lgb.LGBMClassifier(random_state=RND, n_estimators=100, verbose=-1)),
        ],
        voting="soft",
    )

    # Train on full data
    X_train = data_dict["X_full"]
    y_train = data_dict["y_full"]

    logger.info(f"Training ensemble on {len(X_train)} samples...")
    ensemble.fit(X_train, y_train)

    # Save ensemble model
    model_file = save_dir / "ensemble_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(ensemble, f)

    logger.info(f"Saved ensemble model to {model_file}")

    # Save metadata
    metadata = {
        "model_name": "ensemble",
        "model_type": "VotingClassifier",
        "estimators": [
            "LogisticRegression",
            "RandomForestClassifier",
            "LGBMClassifier",
        ],
        "training_samples": len(X_train),
        "features": X_train.columns.tolist(),
        "target_classes": sorted(y_train.unique().tolist()),
    }

    metadata_file = save_dir / "ensemble_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved ensemble metadata to {metadata_file}")


def main():
    """Main function to train and save all models."""
    setup_logging()
    logger = get_logger(__name__)

    logger.info("🚀 Starting model training and saving process...")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load and prepare data
    logger.info("📊 Loading and preparing data...")
    df_tr, df_te, submission = load_data_with_external_merge()

    # Preprocess data (prep function expects target column in df_tr)
    X_train, X_test, y_train, label_encoder = prep(df_tr, df_te)

    # For full data, we use all available training data
    X_full = X_train.copy()
    y_full = y_train.copy()

    data_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "X_full": X_full,
        "y_full": y_full,
    }

    # Train and save individual stacks
    stack_names = ["A", "B", "C", "D", "E", "F"]

    for stack_name in stack_names:
        try:
            train_and_save_stack(stack_name, data_dict, models_dir)
        except Exception as e:
            logger.error(f"Failed to train stack {stack_name}: {e}")
            continue

    # Train and save ensemble model
    try:
        train_ensemble_model(data_dict, models_dir)
    except Exception as e:
        logger.error(f"Failed to train ensemble model: {e}")

    logger.info("✅ Model training and saving complete!")
    logger.info(f"Models saved in: {models_dir.absolute()}")


if __name__ == "__main__":
    main()
