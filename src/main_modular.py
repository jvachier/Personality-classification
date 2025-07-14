#!/usr/bin/env python3
"""Six-Stack Personality Classification Pipeline (Modular Version).

Complete implementation with Optuna optimization and integrated MLOps infrastructure.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import optuna
import pandas as pd

# Import all required modules
from modules.config import (
    ENABLE_PSEUDO_LABELLING,
    LABEL_NOISE_RATE,
    N_TRIALS_BLEND,
    N_TRIALS_STACK,
    PSEUDO_CONFIDENCE_THRESHOLD,
    PSEUDO_MAX_RATIO,
    RND,
    TESTING_MODE,
    TESTING_SAMPLE_SIZE,
    setup_logging,
)
from modules.data_augmentation import apply_data_augmentation
from modules.data_loader import load_data_with_external_merge
from modules.ensemble import improved_blend_obj, oof_probs, oof_probs_noisy
from modules.model_builders import (
    build_neural_stack,
    build_noisy_stack,
    build_sklearn_stack,
    build_stack,
    build_stack_c,
)
from modules.optimization import (
    add_label_noise,
    load_best_trial_params,
    make_neural_stack_objective,
    make_noisy_stack_objective,
    make_sklearn_stack_objective,
    make_stack_c_objective,
    make_stack_objective,
    save_best_trial_params,
)
from modules.preprocessing import add_pseudo_labeling_conservative, prep
from modules.utils import get_logger


@dataclass
class StackConfig:
    """Configuration for a single stack in the ensemble."""

    name: str
    display_name: str
    seed: int
    objective_func: str
    sampler_startup_trials: int = 10
    wide_hp: bool | None = None
    noise_rate: float | None = None


class TrainingData(NamedTuple):
    """Container for training data."""

    X_full: pd.DataFrame
    X_test: pd.DataFrame
    y_full: pd.Series
    le: Any  # LabelEncoder from sklearn
    submission: pd.DataFrame


class StackResults(NamedTuple):
    """Container for stack training results."""

    studies: dict[str, optuna.Study]
    builders: dict[str, Callable[[], Any]]
    oof_predictions: dict[str, pd.Series]


# Set up logging
setup_logging()
logger = get_logger(__name__)


def load_and_prepare_data(
    testing_mode: bool = True, test_size: int = 1000
) -> TrainingData:
    """Load and prepare training data."""
    logger.info("🎯 Six-Stack Personality Classification Pipeline (Modular)")
    logger.info("=" * 60)

    # Load data using advanced merge strategy
    df_tr, df_te, submission = load_data_with_external_merge()

    # FOR TESTING: Limit to specified samples for faster execution
    if testing_mode and len(df_tr) > test_size:
        logger.info(
            f"🔬 TESTING MODE: Limiting dataset to {test_size} samples "
            f"(original: {len(df_tr)})"
        )
        df_tr = df_tr.sample(n=test_size, random_state=RND).reset_index(drop=True)
        logger.info(f"   📊 Using {len(df_tr)} samples for testing")

    # Preprocess data with advanced competitive approach (do this first)
    X_full, X_test, y_full, le = prep(df_tr, df_te)

    # Apply new data augmentation after preprocessing
    X_full, y_full = apply_data_augmentation(X_full, y_full)

    return TrainingData(X_full, X_test, y_full, le, submission)


def get_stack_configurations() -> list[StackConfig]:
    """Define configuration for all stacks."""
    return [
        StackConfig(
            name="A",
            display_name="Traditional ML (narrow)",
            seed=RND,
            objective_func="make_stack_objective",
            sampler_startup_trials=5,
            wide_hp=False,
        ),
        StackConfig(
            name="B",
            display_name="Traditional ML (wide)",
            seed=2024,
            objective_func="make_stack_objective",
            sampler_startup_trials=5,
            wide_hp=True,
        ),
        StackConfig(
            name="C",
            display_name="XGBoost + CatBoost",
            seed=1337,
            objective_func="make_stack_c_objective",
            sampler_startup_trials=10,
        ),
        StackConfig(
            name="D",
            display_name="Sklearn models",
            seed=9999,
            objective_func="make_sklearn_stack_objective",
            sampler_startup_trials=10,
        ),
        StackConfig(
            name="E",
            display_name="Neural Networks",
            seed=7777,
            objective_func="make_neural_stack_objective",
            sampler_startup_trials=10,
        ),
        StackConfig(
            name="F",
            display_name="Noisy Labels",
            seed=5555,
            objective_func="make_noisy_stack_objective",
            sampler_startup_trials=10,
            noise_rate=LABEL_NOISE_RATE,
        ),
    ]


def create_optuna_study(config: StackConfig) -> optuna.Study:
    """Create and configure an Optuna study for a stack."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config.sampler_startup_trials
        ),
    )

    # Load and enqueue best parameters if available
    best_params = load_best_trial_params(f"stack_{config.name}")
    if best_params:
        study.enqueue_trial(best_params)

    return study


def get_objective_function(config: StackConfig, data: TrainingData):
    """Get the appropriate objective function for a stack configuration."""
    objective_funcs = {
        "make_stack_objective": lambda: make_stack_objective(
            data.X_full,
            data.y_full,
            seed=config.seed,
            wide_hp=config.wide_hp,
            sample_weights=None,
        ),
        "make_stack_c_objective": lambda: make_stack_c_objective(
            data.X_full, data.y_full, seed=config.seed, sample_weights=None
        ),
        "make_sklearn_stack_objective": lambda: make_sklearn_stack_objective(
            data.X_full, data.y_full, seed=config.seed, sample_weights=None
        ),
        "make_neural_stack_objective": lambda: make_neural_stack_objective(
            data.X_full, data.y_full, seed=config.seed, sample_weights=None
        ),
        "make_noisy_stack_objective": lambda: make_noisy_stack_objective(
            data.X_full,
            data.y_full,
            seed=config.seed,
            noise_rate=config.noise_rate,
            sample_weights=None,
        ),
    }

    return objective_funcs[config.objective_func]()


def train_single_stack(config: StackConfig, data: TrainingData) -> optuna.Study:
    """Train a single stack with the given configuration."""
    logger.info(f"Training Stack {config.name} - {config.display_name}...")

    study = create_optuna_study(config)
    objective_func = get_objective_function(config, data)

    study.optimize(
        objective_func,
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )

    save_best_trial_params(study, f"stack_{config.name}")
    return study


def train_all_stacks(data: TrainingData) -> dict[str, optuna.Study]:
    """Train all stacks in the ensemble."""
    logger.info("\n🔍 Training 6 specialized stacks...")

    stack_configs = get_stack_configurations()
    studies = {}

    for config in stack_configs:
        studies[config.name] = train_single_stack(config, data)

    return studies


def create_model_builders(
    studies: dict[str, optuna.Study], data: TrainingData
) -> dict[str, Callable[[], Any]]:
    """Create model builder functions for each stack."""
    logger.info("\n📊 Creating model builders for ensemble...")

    builders = {
        "A": lambda: build_stack(studies["A"].best_trial, seed=RND, wide_hp=False),
        "B": lambda: build_stack(studies["B"].best_trial, seed=2024, wide_hp=True),
        "C": lambda: build_stack_c(studies["C"].best_trial, seed=1337),
        "D": lambda: build_sklearn_stack(
            studies["D"].best_trial, seed=9999, X_full=data.X_full
        ),
        "E": lambda: build_neural_stack(
            studies["E"].best_trial, seed=7777, X_full=data.X_full
        ),
        "F": lambda: build_noisy_stack(
            studies["F"].best_trial, seed=5555, noise_rate=LABEL_NOISE_RATE
        ),
    }

    return builders


def generate_oof_predictions(
    builders: dict[str, Callable[[], Any]], data: TrainingData
) -> dict[str, pd.Series]:
    """Generate out-of-fold predictions for all stacks."""
    logger.info("\n🔮 Generating out-of-fold predictions...")

    oof_predictions = {}

    # Generate OOF for stacks A-E
    for stack_name in ["A", "B", "C", "D", "E"]:
        logger.info(f"Generating OOF {stack_name}...")
        oof_predictions[stack_name], _ = oof_probs(
            builders[stack_name],
            data.X_full,
            data.y_full,
            data.X_test[:1],
            sample_weights=None,
        )

    # Generate OOF for stack F (with noisy labels)
    logger.info("Generating OOF F...")
    oof_predictions["F"], _ = oof_probs_noisy(
        builders["F"],
        data.X_full,
        data.y_full,
        data.X_test[:1],
        noise_rate=LABEL_NOISE_RATE,
        sample_weights=None,
    )

    return oof_predictions


def create_blend_objective(oof_predictions: dict[str, pd.Series], y_full: pd.Series):
    """Create the blend objective function."""

    def blend_objective(trial):
        return improved_blend_obj(
            trial,
            oof_predictions["A"],
            oof_predictions["B"],
            oof_predictions["C"],
            oof_predictions["D"],
            oof_predictions["E"],
            oof_predictions["F"],
            y_full,
        )

    return blend_objective


def optimize_ensemble_blending(
    oof_predictions: dict[str, pd.Series], y_full: pd.Series
) -> tuple[dict[str, float], float]:
    """Optimize ensemble blending weights."""
    logger.info("\n⚖️ Optimizing ensemble blending...")

    study_blend = optuna.create_study(direction="maximize")
    blend_objective = create_blend_objective(oof_predictions, y_full)

    study_blend.optimize(
        blend_objective, n_trials=N_TRIALS_BLEND, show_progress_bar=True
    )

    # Extract best weights
    best_weights_list = study_blend.best_trial.user_attrs["weights"]
    best_weights = {
        "A": best_weights_list[0],
        "B": best_weights_list[1],
        "C": best_weights_list[2],
        "D": best_weights_list[3],
        "E": best_weights_list[4],
        "F": best_weights_list[5],
    }

    logger.info("\n🏆 Best ensemble weights:")
    for stack_name, weight in best_weights.items():
        logger.info(f"   Stack {stack_name}: {weight:.3f}")
    logger.info(f"Best CV score: {study_blend.best_value:.6f}")

    return best_weights, study_blend.best_value


def refit_and_predict(
    builders: dict[str, Callable[[], Any]],
    best_weights: dict[str, float],
    data: TrainingData,
) -> tuple[pd.DataFrame, str]:
    """Refit models on full data and generate final predictions."""
    logger.info("\n🔄 Refitting models on full data...")

    # Refit all models
    models = {}
    for stack_name in ["A", "B", "C", "D", "E"]:
        logger.info(f"Refitting Stack {stack_name}...")
        models[stack_name] = builders[stack_name]()
        models[stack_name].fit(data.X_full, data.y_full)

    # Refit Stack F with noisy labels
    logger.info("Refitting Stack F (with noisy labels)...")
    y_full_noisy = add_label_noise(
        data.y_full, noise_rate=LABEL_NOISE_RATE, random_state=RND
    )
    models["F"] = builders["F"]()
    models["F"].fit(data.X_full, y_full_noisy)

    # Generate final predictions
    logger.info("\n🎯 Generating final predictions...")
    probabilities = {}
    for stack_name in ["A", "B", "C", "D", "E", "F"]:
        probabilities[stack_name] = models[stack_name].predict_proba(data.X_test)[:, 1]

    # Final weighted predictions
    proba_test_continuous = sum(
        best_weights[stack_name] * probabilities[stack_name]
        for stack_name in ["A", "B", "C", "D", "E", "F"]
    )

    proba_test_discrete = (proba_test_continuous >= 0.5).astype(int)
    personality = data.le.inverse_transform(proba_test_discrete)

    # Create submission
    submission_df = pd.DataFrame({"id": data.submission.id, "Personality": personality})

    # Save results
    output_file = "./submissions/six_stack_personality_predictions_modular.csv"
    submission_df.to_csv(output_file, index=False)

    return submission_df, output_file


def apply_pseudo_labelling(
    builders: dict[str, Callable[[], Any]],
    best_weights: dict[str, float],
    data: TrainingData,
) -> TrainingData:
    """Apply pseudo labelling using ensemble predictions."""
    if not ENABLE_PSEUDO_LABELLING:
        logger.info("🔮 Pseudo labelling disabled")
        return data

    logger.info(
        f"\n🔮 Applying pseudo labelling (threshold={PSEUDO_CONFIDENCE_THRESHOLD}, max_ratio={PSEUDO_MAX_RATIO})..."
    )

    # First train models to get test predictions for pseudo labelling
    logger.info("Training models for pseudo labelling...")
    models = {}

    # Train stacks A-E normally
    for stack_name in ["A", "B", "C", "D", "E"]:
        logger.info(f"Training Stack {stack_name} for pseudo labelling...")
        models[stack_name] = builders[stack_name]()
        models[stack_name].fit(data.X_full, data.y_full)

    # Train Stack F with noisy labels
    logger.info("Training Stack F (with noisy labels) for pseudo labelling...")
    y_full_noisy = add_label_noise(
        data.y_full, noise_rate=LABEL_NOISE_RATE, random_state=RND
    )
    models["F"] = builders["F"]()
    models["F"].fit(data.X_full, y_full_noisy)

    # Generate test predictions for all stacks
    logger.info("Generating test predictions for pseudo labelling...")
    test_probabilities = {}
    for stack_name in ["A", "B", "C", "D", "E", "F"]:
        test_probabilities[stack_name] = models[stack_name].predict_proba(data.X_test)[
            :, 1
        ]

    # Apply pseudo labelling
    X_combined, y_combined, pseudo_stats = add_pseudo_labeling_conservative(
        data.X_full,
        data.y_full,
        data.X_test,
        test_probabilities["A"],
        test_probabilities["B"],
        test_probabilities["C"],
        test_probabilities["D"],
        test_probabilities["E"],
        test_probabilities["F"],
        best_weights["A"],
        best_weights["B"],
        best_weights["C"],
        best_weights["D"],
        best_weights["E"],
        best_weights["F"],
        confidence_threshold=PSEUDO_CONFIDENCE_THRESHOLD,
        max_pseudo_ratio=PSEUDO_MAX_RATIO,
    )

    # Create new TrainingData with pseudo labels added
    if pseudo_stats["n_pseudo_added"] > 0:
        logger.info(
            f"✅ Pseudo labelling added {pseudo_stats['n_pseudo_added']} samples"
        )

        # Create new TrainingData object with enhanced training set
        enhanced_data = TrainingData(
            X_full=X_combined,
            y_full=y_combined,
            X_test=data.X_test,
            le=data.le,
            submission=data.submission,
        )
        return enhanced_data
    else:
        logger.info("⚠️ No pseudo labels added, using original data")
        return data


def main():
    """Main execution function for the Six-Stack Personality Classification Pipeline."""

    logger.info("🚀 Starting Six-Stack Personality Classification Pipeline")

    try:
        # Load and prepare data
        data = load_and_prepare_data(
            testing_mode=TESTING_MODE, test_size=TESTING_SAMPLE_SIZE
        )

        logger.info(
            f"📊 Loaded data: {len(data.X_full)} training samples, {len(data.X_test)} test samples"
        )

        # Train all stacks
        studies = train_all_stacks(data)

        # Log stack optimization results
        for stack_name, study in studies.items():
            logger.info(
                f"📈 Stack {stack_name}: Best score = {study.best_value:.6f} ({len(study.trials)} trials)"
            )

        # Create model builders
        builders = create_model_builders(studies, data)

        # Generate out-of-fold predictions
        oof_predictions = generate_oof_predictions(builders, data)

        # Optimize ensemble blending
        best_weights, best_cv_score = optimize_ensemble_blending(
            oof_predictions, data.y_full
        )

        logger.info(f"🎯 Best ensemble CV score: {best_cv_score:.6f}")
        logger.info(f"⚖️ Ensemble weights: {best_weights}")

        # Apply pseudo labelling using ensemble predictions
        enhanced_data = apply_pseudo_labelling(builders, best_weights, data)

        # Log pseudo labelling results
        if len(enhanced_data.X_full) > len(data.X_full):
            pseudo_added = len(enhanced_data.X_full) - len(data.X_full)
            logger.info(f"🏷️ Added {pseudo_added} pseudo-labeled samples")

        # Refit models and generate final predictions
        submission_df, output_file = refit_and_predict(
            builders, best_weights, enhanced_data
        )

        # Print final results
        logger.info(f"\n✅ Predictions saved to '{output_file}'")
        logger.info(f"📊 Final submission shape: {submission_df.shape}")
        logger.info("🎉 Six-stack ensemble pipeline completed successfully!")

        # Print summary
        logger.info("\n📋 Summary:")
        logger.info(f"   - Training samples: {len(enhanced_data.X_full):,}")
        logger.info(f"   - Test samples: {len(enhanced_data.X_test):,}")
        logger.info(f"   - Features: {enhanced_data.X_full.shape[1]}")
        logger.info("   - Stacks trained: 6 (A-F)")
        logger.info(f"   - Best ensemble CV score: {best_cv_score:.6f}")
        logger.info(
            f"   - Pseudo labelling: {'Enabled' if ENABLE_PSEUDO_LABELLING else 'Disabled'}"
        )
        logger.info("   - Modular architecture")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
