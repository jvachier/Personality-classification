"""
Ensemble functions for out-of-fold predictions and blending optimization.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from .config import N_SPLITS, RND
from .utils import get_logger, add_label_noise

logger = get_logger(__name__)


def oof_probs(builder, X_full, y_full, X_test, sample_weights=None):
    """
    Generate out-of-fold predictions using cross-validation.
    
    Args:
        builder: Function that returns a model pipeline
        X_full: Training features
        y_full: Training labels
        X_test: Test features (not used in training, just for consistent shape)
        sample_weights: Optional sample weights
        
    Returns:
        Tuple of (oof_predictions, test_predictions_placeholder)
    """
    logger.info(f"   📊 Generating OOF predictions with {N_SPLITS}-fold CV...")
    
    # Initialize out-of-fold predictions
    oof_preds = np.zeros(len(X_full))
    
    # Create cross-validation splits
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"      Fold {fold + 1}/{N_SPLITS}")
        
        # Split data
        X_train_fold = X_full.iloc[train_idx]
        y_train_fold = y_full.iloc[train_idx]
        X_val_fold = X_full.iloc[val_idx]
        
        # Build and train model
        model = builder()
        
        # Handle sample weights if provided
        if sample_weights is not None:
            fold_weights = sample_weights[train_idx]
            model.fit(X_train_fold, y_train_fold, sample_weight=fold_weights)
        else:
            model.fit(X_train_fold, y_train_fold)
        
        # Generate predictions for validation set
        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds
    
    # Calculate CV score
    cv_score = accuracy_score(y_full, oof_preds >= 0.5)
    logger.info(f"   ✅ OOF CV accuracy: {cv_score:.6f}")
    
    return oof_preds, np.zeros(len(X_test))  # Placeholder for test predictions


def oof_probs_noisy(builder, X_full, y_full, X_test, noise_rate=0.02, sample_weights=None):
    """
    Generate out-of-fold predictions using cross-validation with noisy labels.
    
    Args:
        builder: Function that returns a model pipeline
        X_full: Training features
        y_full: Training labels
        X_test: Test features (not used in training, just for consistent shape)
        noise_rate: Rate of label noise to add
        sample_weights: Optional sample weights
        
    Returns:
        Tuple of (oof_predictions, test_predictions_placeholder)
    """
    logger.info(f"   📊 Generating OOF predictions with noisy labels (noise: {noise_rate:.1%})...")
    
    # Initialize out-of-fold predictions
    oof_preds = np.zeros(len(X_full))
    
    # Create cross-validation splits
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"      Fold {fold + 1}/{N_SPLITS}")
        
        # Split data
        X_train_fold = X_full.iloc[train_idx]
        y_train_fold = y_full.iloc[train_idx]
        X_val_fold = X_full.iloc[val_idx]
        
        # Add noise to training labels
        y_train_noisy = add_label_noise(y_train_fold, noise_rate=noise_rate, random_state=RND + fold)
        
        # Build and train model
        model = builder()
        
        # Handle sample weights if provided
        if sample_weights is not None:
            fold_weights = sample_weights[train_idx]
            model.fit(X_train_fold, y_train_noisy, sample_weight=fold_weights)
        else:
            model.fit(X_train_fold, y_train_noisy)
        
        # Generate predictions for validation set
        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds
    
    # Calculate CV score (against clean labels)
    cv_score = accuracy_score(y_full, oof_preds >= 0.5)
    logger.info(f"   ✅ Noisy OOF CV accuracy: {cv_score:.6f}")
    
    return oof_preds, np.zeros(len(X_test))  # Placeholder for test predictions


def improved_blend_obj(trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_true):
    """
    Objective function for optimizing blend weights with constraints.
    
    Args:
        trial: Optuna trial object
        oof_A, oof_B, oof_C, oof_D, oof_E, oof_F: Out-of-fold predictions from each model
        y_true: True labels
        
    Returns:
        Weighted ensemble accuracy
    """
    # Sample weights with constraints
    w1 = trial.suggest_float("wA", 0.05, 0.5)
    w2 = trial.suggest_float("wB", 0.05, 0.5) 
    w3 = trial.suggest_float("wC", 0.05, 0.5)
    w4 = trial.suggest_float("wD", 0.05, 0.5)
    w5 = trial.suggest_float("wE", 0.05, 0.5)
    w6 = trial.suggest_float("wF", 0.05, 0.5)
    
    # Normalize weights to sum to 1
    total_weight = w1 + w2 + w3 + w4 + w5 + w6
    w1, w2, w3, w4, w5, w6 = w1/total_weight, w2/total_weight, w3/total_weight, w4/total_weight, w5/total_weight, w6/total_weight
    
    # Calculate weighted ensemble predictions
    ensemble_preds = (
        w1 * oof_A + w2 * oof_B + w3 * oof_C + 
        w4 * oof_D + w5 * oof_E + w6 * oof_F
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, ensemble_preds >= 0.5)
    
    # Store normalized weights in trial attributes
    trial.set_user_attr("weights", [w1, w2, w3, w4, w5, w6])
    
    return accuracy
