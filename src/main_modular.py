#!/usr/bin/env python3
"""
Six-Stack Personality Classification Pipeline (Modular Version)
Complete implementation with Optuna optimization matching the monolithic script exactly.
"""

import optuna
import numpy as np
import pandas as pd

# Import all required modules
from modules.config import (
    setup_logging,
    RND,
    N_TRIALS_STACK,
    N_TRIALS_BLEND,
    LABEL_NOISE_RATE,
)
from modules.data_loader import load_data_with_external_merge
from modules.preprocessing import prep
from modules.data_augmentation import apply_data_augmentation
from modules.optimization import (
    save_best_trial_params,
    load_best_trial_params,
    make_stack_objective,
    make_stack_c_objective,
    make_sklearn_stack_objective,
    make_neural_stack_objective,
    make_noisy_stack_objective,
)
from modules.ensemble import oof_probs, oof_probs_noisy, improved_blend_obj
from modules.utils import get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main execution function matching the monolithic script exactly"""
    logger.info("🎯 Six-Stack Personality Classification Pipeline (Modular)")
    logger.info("=" * 60)

    # Load data using TOP-4 solution merge strategy
    df_tr, df_te, submission = load_data_with_external_merge()

    # FOR TESTING: Limit to 1000 samples for faster execution
    logger.info(
        f"🔬 TESTING MODE: Limiting dataset to 1000 samples (original: {len(df_tr)})"
    )
    if len(df_tr) > 1000:
        df_tr = df_tr.sample(n=1000, random_state=RND).reset_index(drop=True)
        logger.info(f"   📊 Using {len(df_tr)} samples for testing")

    # Preprocess data with TOP-4 solution approach (do this first)
    X_full, X_test, y_full, le = prep(df_tr, df_te)

    # Apply new data augmentation after preprocessing
    X_full, y_full = apply_data_augmentation(X_full, y_full)

    # Set up pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    # Train 6 stacks
    logger.info("\n🔍 Training 6 specialized stacks...")

    # Stack E - Neural networks
    logger.info("Training Stack E - Neural Networks...")
    study_E = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_E = load_best_trial_params("stack_E")
    if best_params_E:
        study_E.enqueue_trial(best_params_E)
    study_E.optimize(
        make_neural_stack_objective(X_full, y_full, seed=7777, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_E, "stack_E")

    # Stack F - Noisy labels
    logger.info("Training Stack F - Noisy Labels...")
    study_F = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_F = load_best_trial_params("stack_F")
    if best_params_F:
        study_F.enqueue_trial(best_params_F)
    study_F.optimize(
        make_noisy_stack_objective(
            X_full,
            y_full,
            seed=5555,
            noise_rate=LABEL_NOISE_RATE,
            sample_weights=None,
        ),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_F, "stack_F")

    # Stack A - Traditional ML (narrow hyperparameters)
    logger.info("Training Stack A...")
    study_A = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    best_params_A = load_best_trial_params("stack_A")
    if best_params_A:
        study_A.enqueue_trial(best_params_A)
    study_A.optimize(
        make_stack_objective(
            X_full, y_full, seed=RND, wide_hp=False, sample_weights=None
        ),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_A, "stack_A")

    # Stack B - Traditional ML (wide hyperparameters)
    logger.info("Training Stack B...")
    study_B = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    best_params_B = load_best_trial_params("stack_B")
    if best_params_B:
        study_B.enqueue_trial(best_params_B)
    study_B.optimize(
        make_stack_objective(
            X_full, y_full, seed=2024, wide_hp=True, sample_weights=None
        ),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_B, "stack_B")

    # Stack C - XGBoost + CatBoost
    logger.info("Training Stack C...")
    study_C = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_C = load_best_trial_params("stack_C")
    if best_params_C:
        study_C.enqueue_trial(best_params_C)
    study_C.optimize(
        make_stack_c_objective(X_full, y_full, seed=1337, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_C, "stack_C")

    # Stack D - Sklearn models
    logger.info("Training Stack D - Sklearn...")
    study_D = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_D = load_best_trial_params("stack_D")
    if best_params_D:
        study_D.enqueue_trial(best_params_D)
    study_D.optimize(
        make_sklearn_stack_objective(X_full, y_full, seed=9999, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(study_D, "stack_D")

    # Create builders for final ensemble
    logger.info("\n📊 Creating model builders for ensemble...")

    def builder_A():
        from modules.model_builders import build_stack

        return build_stack(study_A.best_trial, seed=RND, wide_hp=False)

    def builder_B():
        from modules.model_builders import build_stack

        return build_stack(study_B.best_trial, seed=2024, wide_hp=True)

    def builder_C():
        from modules.model_builders import build_stack_c

        return build_stack_c(study_C.best_trial, seed=1337)

    def builder_D():
        from modules.model_builders import build_sklearn_stack

        return build_sklearn_stack(study_D.best_trial, seed=9999, X_full=X_full)

    def builder_E():
        from modules.model_builders import build_neural_stack

        return build_neural_stack(study_E.best_trial, seed=7777, X_full=X_full)

    def builder_F():
        from modules.model_builders import build_noisy_stack

        return build_noisy_stack(
            study_F.best_trial, seed=5555, noise_rate=LABEL_NOISE_RATE
        )

    # Generate out-of-fold predictions
    logger.info("\n🔮 Generating out-of-fold predictions...")
    logger.info("Generating OOF A...")
    oof_A, _ = oof_probs(builder_A, X_full, y_full, X_test[:1], sample_weights=None)

    logger.info("Generating OOF B...")
    oof_B, _ = oof_probs(builder_B, X_full, y_full, X_test[:1], sample_weights=None)

    logger.info("Generating OOF C...")
    oof_C, _ = oof_probs(builder_C, X_full, y_full, X_test[:1], sample_weights=None)

    logger.info("Generating OOF D...")
    oof_D, _ = oof_probs(builder_D, X_full, y_full, X_test[:1], sample_weights=None)

    logger.info("Generating OOF E...")
    oof_E, _ = oof_probs(builder_E, X_full, y_full, X_test[:1], sample_weights=None)

    logger.info("Generating OOF F...")
    oof_F, _ = oof_probs_noisy(
        builder_F,
        X_full,
        y_full,
        X_test[:1],
        noise_rate=LABEL_NOISE_RATE,
        sample_weights=None,
    )

    # Optimize ensemble blending
    logger.info("\n⚖️ Optimizing ensemble blending...")
    study_blend = optuna.create_study(direction="maximize")

    # Create the objective function with the specific OOF predictions
    blend_objective = lambda trial: improved_blend_obj(
        trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_full
    )

    study_blend.optimize(
        blend_objective, n_trials=N_TRIALS_BLEND, show_progress_bar=True
    )

    # Extract best weights
    best_weights = study_blend.best_trial.user_attrs["weights"]
    wA, wB, wC, wD, wE, wF = best_weights

    logger.info(f"\n🏆 Best ensemble weights:")
    logger.info(f"   Stack A: {wA:.3f}")
    logger.info(f"   Stack B: {wB:.3f}")
    logger.info(f"   Stack C: {wC:.3f}")
    logger.info(f"   Stack D: {wD:.3f}")
    logger.info(f"   Stack E: {wE:.3f}")
    logger.info(f"   Stack F: {wF:.3f}")
    logger.info(f"Best CV score: {study_blend.best_value:.6f}")

    # Refit models on full data
    logger.info("\n🔄 Refitting models on full data...")
    logger.info("Refitting Stack A...")
    mdl_A = builder_A()
    mdl_A.fit(X_full, y_full)

    logger.info("Refitting Stack B...")
    mdl_B = builder_B()
    mdl_B.fit(X_full, y_full)

    logger.info("Refitting Stack C...")
    mdl_C = builder_C()
    mdl_C.fit(X_full, y_full)

    logger.info("Refitting Stack D...")
    mdl_D = builder_D()
    mdl_D.fit(X_full, y_full)

    logger.info("Refitting Stack E...")
    mdl_E = builder_E()
    mdl_E.fit(X_full, y_full)

    logger.info("Refitting Stack F (with noisy labels)...")
    from modules.optimization import add_label_noise

    y_full_noisy = add_label_noise(
        y_full, noise_rate=LABEL_NOISE_RATE, random_state=RND
    )
    mdl_F = builder_F()
    mdl_F.fit(X_full, y_full_noisy)

    # Generate final predictions
    logger.info("\n🎯 Generating final predictions...")
    proba_A = mdl_A.predict_proba(X_test)[:, 1]
    proba_B = mdl_B.predict_proba(X_test)[:, 1]
    proba_C = mdl_C.predict_proba(X_test)[:, 1]
    proba_D = mdl_D.predict_proba(X_test)[:, 1]
    proba_E = mdl_E.predict_proba(X_test)[:, 1]
    proba_F = mdl_F.predict_proba(X_test)[:, 1]

    # Final weighted predictions
    proba_test_continuous = (
        wA * proba_A
        + wB * proba_B
        + wC * proba_C
        + wD * proba_D
        + wE * proba_E
        + wF * proba_F
    )

    proba_test_discrete = (proba_test_continuous >= 0.5).astype(int)
    personality = le.inverse_transform(proba_test_discrete)

    # Create submission
    submission_df = pd.DataFrame({"id": submission.id, "Personality": personality})

    # Save results
    output_file = "./submissions/six_stack_personality_predictions_modular.csv"
    submission_df.to_csv(output_file, index=False)

    logger.info(f"\n✅ Predictions saved to '{output_file}'")
    logger.info(f"📊 Final submission shape: {submission_df.shape}")
    logger.info("🎉 Six-stack ensemble pipeline completed successfully!")

    # Print summary
    logger.info("\n📋 Summary:")
    logger.info(f"   - Training samples: {len(X_full):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    logger.info(f"   - Features: {X_full.shape[1]}")
    logger.info("   - Stacks trained: 6 (A-F)")
    logger.info(f"   - Best ensemble CV score: {study_blend.best_value:.6f}")
    logger.info("   - Modular architecture")


if __name__ == "__main__":
    main()
