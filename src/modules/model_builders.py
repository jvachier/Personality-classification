"""
Model builders for different stacking configurations in the personality classification pipeline.
Updated to match six_stack_personality_classifier.py exactly.
"""

import catboost as cb
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .config import N_JOBS, N_SPLITS, THREAD_COUNT
from .utils import get_logger


logger = get_logger(__name__)


def build_stack(trial, seed: int, wide_hp: bool) -> Pipeline:
    """Build main stacking model with CPU-only configuration - matches six_stack_personality_classifier.py exactly"""

    # Enhanced search ranges for better performance
    if wide_hp:
        n_lo, n_hi = 600, 1200
    else:
        n_lo, n_hi = 500, 1000

    # Enhanced XGBoost parameters - CPU ONLY
    grow_policy = trial.suggest_categorical("xgb_grow", ["depthwise", "lossguide"])

    xgb_params = {
        "tree_method": "hist",  # CPU-only method
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "enable_categorical": True,
        "random_state": seed,
        "n_estimators": trial.suggest_int("xgb_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.25, log=True),
        "max_depth": trial.suggest_int("xgb_d", 5, 12),
        "subsample": trial.suggest_float("xgb_sub", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("xgb_alpha", 0.0001, 2.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 0.5, 10.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 8.0),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 15),
        "grow_policy": grow_policy,
        "max_bin": 256,
        "verbosity": 0,
        "n_jobs": N_JOBS,  # Use centralized CPU configuration
    }

    if grow_policy == "lossguide":
        xgb_params["max_leaves"] = trial.suggest_int("xgb_leaves", 50, 200)

    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # Enhanced LightGBM parameters - CPU ONLY
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="cpu",
        verbose=-1,
        random_state=seed,
        n_estimators=trial.suggest_int("lgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.25, log=True),
        max_depth=trial.suggest_int("lgb_d", -1, 15),
        subsample=trial.suggest_float("lgb_sub", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("lgb_col", 0.5, 1.0),
        num_leaves=trial.suggest_int("lgb_leaves", 50, 200),
        min_child_samples=trial.suggest_int("lgb_min_child", 5, 80),
        min_child_weight=trial.suggest_float("lgb_min_weight", 1e-4, 20.0, log=True),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-4, 20.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-4, 20.0, log=True),
        cat_smooth=trial.suggest_int("lgb_cat_smooth", 1, 150),
        cat_l2=trial.suggest_float("lgb_cat_l2", 0.5, 15.0),
        max_bin=255,
        min_data_in_bin=trial.suggest_int("lgb_min_data_bin", 1, 30),
        boost_from_average=True,
        force_row_wise=True,
        path_smooth=trial.suggest_float("lgb_path_smooth", 0, 0.15),
        n_jobs=N_JOBS,
    )

    # Enhanced CatBoost parameters - CPU ONLY
    bootstrap_type = trial.suggest_categorical(
        "cat_bootstrap", ["Bayesian", "Bernoulli", "MVS"]
    )

    cat_params = {
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_state": seed,
        "iterations": trial.suggest_int("cat_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.25, log=True),
        "depth": trial.suggest_int("cat_d", 5, 12),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 0.5, 20.0),
        "random_strength": trial.suggest_float("cat_rs", 0.1, 20.0),
        "leaf_estimation_iterations": trial.suggest_int("cat_leaf_iters", 1, 15),
        "grow_policy": trial.suggest_categorical(
            "cat_grow", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "min_data_in_leaf": trial.suggest_int("cat_min_data", 1, 30),
        "bootstrap_type": bootstrap_type,
        "border_count": trial.suggest_int("cat_border_count", 200, 300),
        "verbose": False,
        "thread_count": THREAD_COUNT,
    }

    if bootstrap_type == "Bayesian":
        cat_params["bagging_temperature"] = trial.suggest_float("cat_temp", 0.0, 1.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        cat_params["subsample"] = trial.suggest_float("cat_subsample", 0.5, 1.0)

    cat_clf = cb.CatBoostClassifier(**cat_params)

    # Enhanced meta-learner
    meta_type = trial.suggest_categorical("meta_type", ["logistic", "ridge", "xgb"])

    if meta_type == "logistic":
        meta = LogisticRegression(
            C=trial.suggest_float("meta_log_c", 0.1, 10.0, log=True),
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    elif meta_type == "ridge":
        meta = LogisticRegression(
            C=1.0
            / trial.suggest_float(
                "meta_ridge_alpha", 0.1, 10.0, log=True
            ),  # C = 1/alpha
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    else:  # xgb
        meta = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=N_JOBS,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=N_JOBS,
    )

    return Pipeline([("stk", stk)])


def build_stack_c(trial, seed: int) -> Pipeline:
    """Build Stack C with XGBoost + CatBoost combination for diversity - matches original exactly."""

    # XGBoost parameters - CPU ONLY
    grow_policy = trial.suggest_categorical("xgb_grow", ["depthwise", "lossguide"])
    xgb_params = {
        "tree_method": "hist",  # CPU-only method
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "enable_categorical": True,
        "random_state": seed,
        "n_estimators": trial.suggest_int("xgb_n", 400, 800),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("xgb_d", 6, 12),
        "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_col", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("xgb_alpha", 0.001, 1.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
        "grow_policy": grow_policy,
        "max_bin": 255,
        "verbosity": 0,
        "n_jobs": N_JOBS,
    }

    if grow_policy == "lossguide":
        xgb_params["max_leaves"] = trial.suggest_int("xgb_leaves", 50, 150)

    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # CatBoost parameters - CPU ONLY
    bootstrap_type = trial.suggest_categorical(
        "cat_bootstrap", ["Bayesian", "Bernoulli", "MVS"]
    )
    cat_params = {
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_state": seed,
        "iterations": trial.suggest_int("cat_n", 400, 800),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("cat_d", 6, 12),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 1.0, 15.0),
        "random_strength": trial.suggest_float("cat_rs", 1.0, 15.0),
        "leaf_estimation_iterations": trial.suggest_int("cat_leaf_iters", 5, 15),
        "grow_policy": trial.suggest_categorical(
            "cat_grow", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "min_data_in_leaf": trial.suggest_int("cat_min_data", 1, 20),
        "bootstrap_type": bootstrap_type,
        "border_count": trial.suggest_int("cat_border_count", 200, 300),
        "verbose": False,
        "thread_count": THREAD_COUNT,
    }

    if bootstrap_type == "Bayesian":
        cat_params["bagging_temperature"] = trial.suggest_float("cat_temp", 0.1, 1.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        cat_params["subsample"] = trial.suggest_float("cat_subsample", 0.6, 0.95)

    cat_clf = cb.CatBoostClassifier(**cat_params)

    # Enhanced meta-learner
    meta_type = trial.suggest_categorical("c_meta_type", ["logistic", "ridge", "xgb"])

    if meta_type == "logistic":
        meta = LogisticRegression(
            C=trial.suggest_float("c_meta_c", 0.1, 10.0, log=True),
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    elif meta_type == "ridge":
        meta = LogisticRegression(
            C=1.0
            / trial.suggest_float("c_meta_alpha", 0.1, 10.0, log=True),  # C = 1/alpha
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    else:  # xgb
        meta = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=N_JOBS,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=N_JOBS,
    )

    return Pipeline([("stk", stk)])


def build_sklearn_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build Stack D with sklearn models with improved preprocessing - matches original exactly."""
    # Since we're using one-hot encoding in preprocessing, we treat all features as numerical
    # Get all columns as numerical (they're all numerical after one-hot encoding)
    num_base = list(X_full.columns)

    # Use RobustScaler for all features since they're all numerical after one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_base),
        ]
    )

    # RandomForest parameters
    rf_clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("rf_n", 500, 1000),
        max_depth=trial.suggest_int("rf_depth", 15, 40),
        min_samples_split=trial.suggest_int("rf_min_split", 2, 10),
        min_samples_leaf=trial.suggest_int("rf_min_leaf", 1, 5),
        max_features=trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", None]
        ),
        bootstrap=True,
        class_weight=trial.suggest_categorical("rf_class_weight", [None, "balanced"]),
        random_state=seed,
        n_jobs=N_JOBS,
    )

    # ExtraTrees parameters
    et_clf = ExtraTreesClassifier(
        n_estimators=trial.suggest_int("et_n", 500, 1000),
        max_depth=trial.suggest_int("et_depth", 15, 40),
        min_samples_split=trial.suggest_int("et_min_split", 2, 10),
        min_samples_leaf=trial.suggest_int("et_min_leaf", 1, 5),
        max_features=trial.suggest_categorical(
            "et_max_features", ["sqrt", "log2", None]
        ),
        bootstrap=False,
        class_weight=trial.suggest_categorical("et_class_weight", [None, "balanced"]),
        random_state=seed,
        n_jobs=N_JOBS,
    )

    # HistGradientBoosting parameters
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("hgb_n", 500, 1000),
        learning_rate=trial.suggest_float("hgb_lr", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("hgb_depth", 8, 20),
        min_samples_leaf=trial.suggest_int("hgb_min_leaf", 5, 30),
        l2_regularization=trial.suggest_float("hgb_l2", 0.0, 2.0),
        random_state=seed,
    )

    # Meta-learner options
    meta_type = trial.suggest_categorical(
        "meta_type", ["logistic", "xgb", "lgb", "ridge"]
    )

    if meta_type == "logistic":
        meta = LogisticRegression(
            C=trial.suggest_float("meta_log_c", 0.1, 10.0, log=True),
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    elif meta_type == "xgb":
        meta = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("meta_xgb_n", 100, 300),
            learning_rate=trial.suggest_float("meta_xgb_lr", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("meta_xgb_depth", 3, 8),
            random_state=seed,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=N_JOBS,
        )
    elif meta_type == "lgb":
        meta = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("meta_lgb_n", 100, 300),
            learning_rate=trial.suggest_float("meta_lgb_lr", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("meta_lgb_depth", 3, 8),
            random_state=seed,
            objective="binary",
            verbose=-1,
            n_jobs=N_JOBS,
        )
    else:  # ridge - use LogisticRegression with L2 for probability support
        meta = LogisticRegression(
            C=1.0
            / trial.suggest_float(
                "meta_ridge_alpha", 0.1, 10.0, log=True
            ),  # C = 1/alpha
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf), ("hgb", hgb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=N_JOBS,
    )

    return Pipeline([("preprocessor", preprocessor), ("stk", stk)])


def build_neural_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build Neural Network Stack with diverse architectures - matches original exactly."""
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
            n_jobs=N_JOBS,
        )
    else:  # ridge
        meta = LogisticRegression(
            C=1.0 / trial.suggest_float("neural_meta_alpha", 0.1, 10.0, log=True),
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=N_JOBS,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("mlp1", mlp1), ("mlp2", mlp2), ("svm", svm_clf), ("nb", nb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=N_JOBS,
    )

    return Pipeline([("preprocessor", preprocessor), ("stk", stk)])


def build_noisy_stack(trial, seed: int, noise_rate: float = 0.02) -> Pipeline:
    """Build a stack trained on noisy labels for regularization - matches original exactly."""
    # Same parameters as build_stack but with different seed for noise
    # Note: noise_rate parameter kept for API compatibility (noise applied externally)
    _ = noise_rate  # Suppress unused argument warning
    n_lo, n_hi = 500, 1000

    # XGBoost with slightly different config for noise robustness - CPU ONLY
    grow_policy = trial.suggest_categorical("xgb_grow", ["depthwise", "lossguide"])

    xgb_params = {
        "tree_method": "hist",  # CPU-only method
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "enable_categorical": True,
        "random_state": seed,
        "n_estimators": trial.suggest_int("xgb_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("xgb_d", 4, 10),
        "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_col", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("xgb_alpha", 0.001, 3.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 15.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("xgb_min_child", 2, 20),
        "grow_policy": grow_policy,
        "max_bin": 256,
        "verbosity": 0,
        "n_jobs": N_JOBS,
    }

    if grow_policy == "lossguide":
        xgb_params["max_leaves"] = trial.suggest_int("xgb_leaves", 31, 150)

    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # LightGBM with noise robustness - CPU ONLY
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="cpu",
        verbose=-1,
        random_state=seed,
        n_estimators=trial.suggest_int("lgb_n", n_lo, n_hi),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("lgb_d", -1, 12),
        subsample=trial.suggest_float("lgb_sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("lgb_col", 0.6, 1.0),
        num_leaves=trial.suggest_int("lgb_leaves", 31, 150),
        min_child_samples=trial.suggest_int("lgb_min_child", 10, 100),
        min_child_weight=trial.suggest_float("lgb_min_weight", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 30.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-3, 30.0, log=True),
        cat_smooth=trial.suggest_int("lgb_cat_smooth", 10, 200),
        cat_l2=trial.suggest_float("lgb_cat_l2", 1.0, 20.0),
        max_bin=255,
        min_data_in_bin=trial.suggest_int("lgb_min_data_bin", 3, 50),
        boost_from_average=True,
        force_row_wise=True,
        path_smooth=trial.suggest_float("lgb_path_smooth", 0, 0.2),
        n_jobs=N_JOBS,
    )

    # CatBoost with noise robustness - CPU ONLY
    bootstrap_type = trial.suggest_categorical(
        "cat_bootstrap", ["Bayesian", "Bernoulli", "MVS"]
    )

    cat_params = {
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_state": seed,
        "iterations": trial.suggest_int("cat_n", n_lo, n_hi),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("cat_d", 4, 10),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 1.0, 25.0),
        "random_strength": trial.suggest_float("cat_rs", 1.0, 25.0),
        "leaf_estimation_iterations": trial.suggest_int("cat_leaf_iters", 1, 20),
        "grow_policy": trial.suggest_categorical(
            "cat_grow", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "min_data_in_leaf": trial.suggest_int("cat_min_data", 5, 50),
        "bootstrap_type": bootstrap_type,
        "border_count": trial.suggest_int("cat_border_count", 150, 300),
        "verbose": False,
        "thread_count": THREAD_COUNT,
    }

    if bootstrap_type == "Bayesian":
        cat_params["bagging_temperature"] = trial.suggest_float("cat_temp", 0.0, 1.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        cat_params["subsample"] = trial.suggest_float("cat_subsample", 0.5, 1.0)

    cat_clf = cb.CatBoostClassifier(**cat_params)

    # Enhanced meta-learner for noisy labels
    meta = LogisticRegression(
        C=trial.suggest_float("meta_log_c", 0.01, 2.0, log=True),
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
        n_jobs=N_JOBS,
    )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=N_JOBS,
    )

    return Pipeline([("stk", stk)])
