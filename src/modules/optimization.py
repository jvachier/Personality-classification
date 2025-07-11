"""
Optimization functions and parameter utilities for the personality classification pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from .utils import get_logger
from .config import N_SPLITS
from .model_builders import (
    build_stack,
    build_stack_c,
    build_sklearn_stack,
    build_neural_stack,
    build_noisy_stack,
)

logger = get_logger(__name__)


def save_best_trial_params(study, model_name, params_dir="best_params"):
    """Save the best trial parameters to a JSON file."""
    os.makedirs(params_dir, exist_ok=True)
    best_params = study.best_trial.params
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    with open(filepath, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters for {model_name} to {filepath}")
    return best_params


def load_best_trial_params(model_name, params_dir="best_params"):
    """Load the best trial parameters from a JSON file."""
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            params = json.load(f)
        logger.info(f"Loaded best parameters for {model_name} from {filepath}")
        return params
    else:
        logger.info(f"No saved parameters found for {model_name} at {filepath}")
        return None


def add_label_noise(y, noise_rate=0.02, random_state=42):
    """
    Add controlled label noise for regularization.
    """
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_indices = np.random.choice(len(y), n_flip, replace=False)

    # Flip labels (0->1, 1->0)
    y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]

    return y_noisy


def make_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, wide_hp: bool, sample_weights=None
):
    """Create an objective function for Optuna to optimize stacking models."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_stack(trial, seed=seed, wide_hp=wide_hp)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_stack_c_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for Stack C (XGBoost + CatBoost)."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_stack_c(trial, seed=seed)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_sklearn_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for sklearn-based Stack D."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_sklearn_stack(trial, seed=seed, X_full=X)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_neural_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for neural network Stack E."""

    def _obj(trial):
        try:
            # Build the neural network stacking model
            model = build_neural_stack(trial, seed=seed, X_full=X)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_noisy_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, noise_rate: float, sample_weights=None
):
    """Create an objective function for noisy label Stack F."""

    def _obj(trial):
        try:
            # Build the noisy stacking model
            model = build_noisy_stack(trial, seed=seed, noise_rate=noise_rate)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Add label noise to training data
                y_train_noisy = add_label_noise(
                    y_train, noise_rate=noise_rate, random_state=seed + fold
                )

                # Fit model
                model.fit(X_train, y_train_noisy)

                # Predict and score (on clean validation labels)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj
