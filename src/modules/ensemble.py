"""
Ensemble functions for out-of-fold predictions and blending optimization.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .config import N_SPLITS, RND
from .optimization import add_label_noise
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class NoisyEnsembleConfig:
    """Configuration for noisy label ensemble training."""
    model_builder: Callable
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    noise_rate: float
    sample_weights: np.ndarray | None = None


def oof_probs(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    sample_weights=None,
):
    """Generate out-of-fold predictions for ensemble blending."""
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info(f"   Fold {fold + 1}/{N_SPLITS}")

        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, _y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Build and fit model
        model = model_builder()
        model.fit(X_train, y_train)

        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds


def oof_probs_noisy(config: NoisyEnsembleConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for noisy label ensemble.

    Args:
        config: Configuration containing model builder, data, and noise parameters

    Returns:
        Tuple of (oof_predictions, test_predictions)
    """
    oof_preds = np.zeros(len(config.X))
    test_preds = np.zeros(len(config.X_test))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(config.X, config.y)):
        logger.info(
            f"   Fold {fold + 1}/{N_SPLITS} (with {config.noise_rate:.1%} label noise)"
        )

        X_train, X_val = config.X.iloc[tr_idx], config.X.iloc[val_idx]
        y_train, _y_val = config.y.iloc[tr_idx], config.y.iloc[val_idx]

        # Add label noise to training data
        y_train_noisy = add_label_noise(
            y_train, noise_rate=config.noise_rate, random_state=RND + fold
        )

        # Build and fit model
        model = config.model_builder()
        model.fit(X_train, y_train_noisy)

        # Out-of-fold predictions (on clean validation data)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(config.X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds


def improved_blend_obj(trial, oof_predictions: dict[str, np.ndarray], y_true):
    """Improved blending objective with constraints and regularization.

    Args:
        trial: Optuna trial object
        oof_predictions: Dictionary mapping stack names to their OOF predictions
        y_true: True labels
    """
    # Sample blend weights for each stack
    weights = {}
    for stack_name in oof_predictions:
        weights[stack_name] = trial.suggest_float(f"w_{stack_name}", 0.0, 1.0)

    # Normalize weights
    weight_values = np.array(list(weights.values()))
    normalized_weights = weight_values / np.sum(weight_values)

    # Calculate blended predictions
    blended = np.zeros_like(y_true, dtype=float)
    for i, (_stack_name, oof_pred) in enumerate(oof_predictions.items()):
        blended += normalized_weights[i] * oof_pred

    # Convert to binary predictions
    y_pred = (blended >= 0.5).astype(int)

    # Calculate accuracy
    score = accuracy_score(y_true, y_pred)

    # Store normalized weights in trial attributes
    trial.set_user_attr("weights", normalized_weights.tolist())
    trial.set_user_attr("stack_names", list(oof_predictions.keys()))

    return score
