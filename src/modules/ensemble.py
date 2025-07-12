"""
Ensemble functions for out-of-fold predictions and blending optimization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .config import N_SPLITS, RND
from .optimization import add_label_noise
from .utils import get_logger

logger = get_logger(__name__)


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


def oof_probs_noisy(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    noise_rate: float,
    sample_weights=None,
):
    """Generate out-of-fold predictions for noisy label ensemble."""
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info(
            f"   Fold {fold + 1}/{N_SPLITS} (with {noise_rate:.1%} label noise)"
        )

        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, _y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Add label noise to training data
        y_train_noisy = add_label_noise(
            y_train, noise_rate=noise_rate, random_state=RND + fold
        )

        # Build and fit model
        model = model_builder()
        model.fit(X_train, y_train_noisy)

        # Out-of-fold predictions (on clean validation data)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds


def improved_blend_obj(trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_true):
    """Improved blending objective with constraints and regularization."""
    # Sample blend weights
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0)
    w3 = trial.suggest_float("w3", 0.0, 1.0)
    w4 = trial.suggest_float("w4", 0.0, 1.0)
    w5 = trial.suggest_float("w5", 0.0, 1.0)
    w6 = trial.suggest_float("w6", 0.0, 1.0)

    # Normalize weights
    weights = np.array([w1, w2, w3, w4, w5, w6])
    weights = weights / np.sum(weights)

    # Calculate blended predictions
    blended = (
        weights[0] * oof_A
        + weights[1] * oof_B
        + weights[2] * oof_C
        + weights[3] * oof_D
        + weights[4] * oof_E
        + weights[5] * oof_F
    )

    # Convert to binary predictions
    y_pred = (blended >= 0.5).astype(int)

    # Calculate accuracy
    score = accuracy_score(y_true, y_pred)

    # Store normalized weights in trial attributes
    trial.set_user_attr("weights", weights.tolist())

    return score
