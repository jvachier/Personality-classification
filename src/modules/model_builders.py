"""
Model builders for different stacking configurations in the personality classification pipeline.
"""

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from .config import N_SPLITS
from .utils import get_logger

logger = get_logger(__name__)


def build_stack(trial, seed: int, wide_hp: bool) -> Pipeline:
    """Build simplified main stacking model for stability"""
    # Reduced search ranges for stability
    if wide_hp:
        n_lo, n_hi = 150, 350  # Reduced from 600-1200
    else:
        n_lo, n_hi = 100, 250  # Reduced from 500-1000

    # Simplified XGBoost parameters - CPU ONLY
    xgb_params = {
        "tree_method": "hist",
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "random_state": seed,
        "n_estimators": trial.suggest_int("xgb_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("xgb_lr", 0.05, 0.15),  # Narrower range
        "max_depth": trial.suggest_int("xgb_d", 4, 8),  # Reduced depth
        "subsample": trial.suggest_float("xgb_sub", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("xgb_col", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("xgb_alpha", 0.01, 2.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 5.0),
        "verbosity": 0,
        "n_jobs": 1,  # Single thread for stability
    }

    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # Simplified LightGBM parameters - CPU ONLY
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="cpu",
        verbose=-1,
        random_state=seed,
        n_estimators=trial.suggest_int("lgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("lgb_lr", 0.05, 0.15),
        max_depth=trial.suggest_int("lgb_d", 4, 8),  # Reduced depth
        subsample=trial.suggest_float("lgb_sub", 0.7, 0.9),
        colsample_bytree=trial.suggest_float("lgb_col", 0.7, 0.9),
        num_leaves=trial.suggest_int("lgb_leaves", 31, 100),  # Reduced leaves
        min_child_samples=trial.suggest_int("lgb_min_child", 10, 50),
        min_child_weight=trial.suggest_float("lgb_min_weight", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-3, 10.0, log=True),
        max_bin=255,
        boost_from_average=True,
        force_row_wise=True,
        n_jobs=1,  # Single thread for stability
    )

    # Simplified meta-learner (remove complex CatBoost)
    meta = LogisticRegression(
        C=trial.suggest_float("meta_log_c", 0.1, 2.0),
        max_iter=1000,  # Reduced iterations
        solver="lbfgs",
        random_state=seed,
        n_jobs=1,  # Single thread
    )

    skf_inner = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed
    )  # Reduced folds

    # Use only 2 models instead of 3 for stability (removed CatBoost)
    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=1,  # Single thread for stability
    )

    return Pipeline([("stk", stk)])


def build_stack_c(trial, seed: int) -> Pipeline:
    """Build simplified Stack C with XGBoost + LightGBM for stability."""
    # Reduced estimators for stability
    n_lo, n_hi = 100, 250

    # Simplified XGBoost parameters - CPU ONLY
    xgb_params = {
        "tree_method": "hist",
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "random_state": seed,
        "n_estimators": trial.suggest_int("xgb_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("xgb_lr", 0.05, 0.15),
        "max_depth": trial.suggest_int("xgb_d", 4, 8),  # Reduced depth
        "subsample": trial.suggest_float("xgb_sub", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("xgb_col", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("xgb_alpha", 0.01, 1.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 5.0),
        "verbosity": 0,
        "n_jobs": 1,  # Single thread for stability
    }

    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # Use LightGBM instead of CatBoost for better stability
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="cpu",
        verbose=-1,
        random_state=seed,
        n_estimators=trial.suggest_int("lgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("lgb_lr", 0.05, 0.15),
        max_depth=trial.suggest_int("lgb_d", 4, 8),
        subsample=trial.suggest_float("lgb_sub", 0.7, 0.9),
        colsample_bytree=trial.suggest_float("lgb_col", 0.7, 0.9),
        num_leaves=trial.suggest_int("lgb_leaves", 31, 100),
        min_child_samples=trial.suggest_int("lgb_min_child", 10, 50),
        min_child_weight=trial.suggest_float("lgb_min_weight", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-3, 10.0, log=True),
        max_bin=255,
        boost_from_average=True,
        force_row_wise=True,
        n_jobs=1,
    )

    # Simplified meta-learner
    meta = LogisticRegression(
        C=trial.suggest_float("c_meta_c", 0.1, 2.0),
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
        n_jobs=1,
    )

    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # Use XGBoost + LightGBM instead of XGBoost + CatBoost for stability
    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=1,
    )

    return Pipeline([("stk", stk)])


def build_sklearn_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build simplified Stack D with sklearn models for stability."""
    # Get all columns as numerical (they're all numerical after one-hot encoding)
    num_base = list(X_full.columns)

    # Use RobustScaler for all features since they're all numerical after one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_base),
        ]
    )

    # Simplified RandomForest parameters
    rf_clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("rf_n", 100, 300),  # Reduced from 500-1000
        max_depth=trial.suggest_int("rf_depth", 10, 20),  # Reduced depth
        min_samples_split=trial.suggest_int("rf_min_split", 2, 5),
        min_samples_leaf=trial.suggest_int("rf_min_leaf", 1, 3),
        max_features="sqrt",  # Fixed instead of suggesting
        bootstrap=True,
        random_state=seed,
        n_jobs=1,  # Single thread for stability
    )

    # Simplified ExtraTrees parameters
    et_clf = ExtraTreesClassifier(
        n_estimators=trial.suggest_int("et_n", 100, 300),
        max_depth=trial.suggest_int("et_depth", 10, 20),
        min_samples_split=trial.suggest_int("et_min_split", 2, 5),
        min_samples_leaf=trial.suggest_int("et_min_leaf", 1, 3),
        max_features="sqrt",  # Fixed instead of suggesting
        bootstrap=False,
        random_state=seed,
        n_jobs=1,  # Single thread for stability
    )

    # Simplified meta-learner (remove complex options)
    meta = LogisticRegression(
        C=trial.suggest_float("meta_log_c", 0.1, 2.0),
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
        n_jobs=1,
    )

    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # Use only 2 models instead of 3 for stability (removed HistGradientBoosting)
    stk = StackingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=1,
    )

    return Pipeline([("preprocessor", preprocessor), ("stk", stk)])


def build_neural_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build Neural Network Stack with diverse architectures."""
    # Since we already use one-hot encoding in preprocessing, all features are numerical
    num_base = list(X_full.columns)

    # Use RobustScaler for all features since they're all numerical after preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_base),
        ]
    )

    # MLPClassifier 1 - Deep network
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(
            trial.suggest_int("mlp1_h1", 50, 200),
            trial.suggest_int("mlp1_h2", 20, 100),
            trial.suggest_int("mlp1_h3", 10, 50),
        ),
        learning_rate_init=trial.suggest_float("mlp1_lr", 0.0001, 0.01, log=True),
        alpha=trial.suggest_float("mlp1_alpha", 1e-6, 1e-1, log=True),
        max_iter=trial.suggest_int("mlp1_iter", 500, 2000),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )

    # MLPClassifier 2 - Wide network
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(trial.suggest_int("mlp2_h1", 100, 400),),
        learning_rate_init=trial.suggest_float("mlp2_lr", 0.0001, 0.01, log=True),
        alpha=trial.suggest_float("mlp2_alpha", 1e-6, 1e-1, log=True),
        max_iter=trial.suggest_int("mlp2_iter", 500, 2000),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )

    # SVM with probability estimates
    svm_clf = SVC(
        C=trial.suggest_float("svm_c", 0.1, 10.0, log=True),
        gamma=trial.suggest_categorical("svm_gamma", ["scale", "auto"]),
        kernel=trial.suggest_categorical("svm_kernel", ["rbf", "poly", "sigmoid"]),
        probability=True,
        random_state=seed,
    )

    # Naive Bayes
    nb_clf = GaussianNB(
        var_smoothing=trial.suggest_float("nb_var_smooth", 1e-11, 1e-7, log=True)
    )

    # Meta-learner options
    meta_type = trial.suggest_categorical("neural_meta_type", ["logistic", "ridge"])

    if meta_type == "logistic":
        meta = LogisticRegression(
            C=trial.suggest_float("neural_meta_c", 0.1, 10.0, log=True),
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=4,
        )
    else:  # ridge
        meta = LogisticRegression(
            C=1.0 / trial.suggest_float("neural_meta_alpha", 0.1, 10.0, log=True),
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=4,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("mlp1", mlp1), ("mlp2", mlp2), ("svm", svm_clf), ("nb", nb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=4,
    )

    return Pipeline([("preprocessor", preprocessor), ("stk", stk)])


def build_noisy_stack(trial, seed: int, noise_rate: float = 0.02) -> Pipeline:
    """Build a simplified stack trained on noisy labels for regularization."""
    # Simplified parameters for stability
    n_lo, n_hi = 100, 300  # Reduced from 500-1000

    # Use only LightGBM for stability - lighter than XGBoost+LightGBM+CatBoost
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="cpu",
        verbose=-1,
        random_state=seed,
        n_estimators=trial.suggest_int("lgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("lgb_lr", 0.05, 0.15),  # Narrower range
        max_depth=trial.suggest_int("lgb_d", 4, 8),  # Reduced depth
        subsample=trial.suggest_float("lgb_sub", 0.7, 0.9),
        colsample_bytree=trial.suggest_float("lgb_col", 0.7, 0.9),
        num_leaves=trial.suggest_int("lgb_leaves", 31, 100),  # Reduced leaves
        min_child_samples=trial.suggest_int("lgb_min_child", 10, 50),
        min_child_weight=trial.suggest_float("lgb_min_weight", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-3, 10.0, log=True),
        max_bin=255,
        boost_from_average=True,
        force_row_wise=True,
        n_jobs=1,  # Single thread for stability
    )

    # Simplified XGBoost - reduced complexity
    xgb_clf = xgb.XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        objective="binary:logistic",
        random_state=seed,
        n_estimators=trial.suggest_int("xgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("xgb_lr", 0.05, 0.15),
        max_depth=trial.suggest_int("xgb_d", 4, 8),  # Reduced depth
        subsample=trial.suggest_float("xgb_sub", 0.7, 0.9),
        colsample_bytree=trial.suggest_float("xgb_col", 0.7, 0.9),
        reg_alpha=trial.suggest_float("xgb_alpha", 0.01, 3.0, log=True),
        reg_lambda=trial.suggest_float("xgb_lambda", 1.0, 10.0),
        verbosity=0,
        n_jobs=1,  # Single thread for stability
    )

    # Simplified meta-learner
    meta = LogisticRegression(
        C=trial.suggest_float("meta_log_c", 0.1, 1.0),
        max_iter=1000,  # Reduced iterations
        solver="lbfgs",
        random_state=seed,
        n_jobs=1,  # Single thread
    )

    skf_inner = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed
    )  # Reduced folds

    # Use only 2 base models instead of 3 for stability
    stk = StackingClassifier(
        estimators=[("lgb", lgb_clf), ("xgb", xgb_clf)],  # Removed CatBoost
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=1,  # Single thread for stability
    )

    return Pipeline([("stk", stk)])
