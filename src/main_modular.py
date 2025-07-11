#!/usr/bin/env python3
"""
Six-Stack Personality Classification Pipeline (Refactored)
Modular version of the personality classification system
"""

import optuna

# Import our custom modules
from modules.config import setup_logging, N_TRIALS_STACK, N_TRIALS_BLEND, LABEL_NOISE_RATE, RND
from modules.data_loader import load_data_with_external_merge
from modules.preprocessing import prep, add_pseudo_labeling_conservative
from modules.data_augmentation import apply_data_augmentation
from modules.model_builders import (
    build_stack, build_stack_c, build_sklearn_stack, 
    build_neural_stack, build_noisy_stack
)
from modules.optimization import save_best_trial_params, load_best_trial_params
from modules.ensemble import oof_probs, oof_probs_noisy, improved_blend_obj
from modules.utils import add_label_noise

# Set up logging
logger = setup_logging()


def create_objective_functions(X_full, y_full):
    """Create objective functions for different stacks."""
    
    def make_stack_objective(seed, wide_hp, sample_weights=None):
        def objective(trial):
            try:
                # Generate out-of-fold predictions
                oof_preds, _ = oof_probs(
                    lambda: build_stack(trial, seed, wide_hp),
                    X_full, y_full, X_full[:1], sample_weights
                )
                
                # Calculate accuracy
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_full, oof_preds >= 0.5)
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        return objective
    
    def make_stack_c_objective(seed, sample_weights=None):
        def objective(trial):
            try:
                oof_preds, _ = oof_probs(
                    lambda: build_stack_c(trial, seed),
                    X_full, y_full, X_full[:1], sample_weights
                )
                
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_full, oof_preds >= 0.5)
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        return objective
    
    def make_sklearn_stack_objective(seed, sample_weights=None):
        def objective(trial):
            try:
                oof_preds, _ = oof_probs(
                    lambda: build_sklearn_stack(trial, seed, X_full),
                    X_full, y_full, X_full[:1], sample_weights
                )
                
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_full, oof_preds >= 0.5)
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        return objective
    
    def make_neural_stack_objective(seed, sample_weights=None):
        def objective(trial):
            try:
                oof_preds, _ = oof_probs(
                    lambda: build_neural_stack(trial, seed, X_full),
                    X_full, y_full, X_full[:1], sample_weights
                )
                
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_full, oof_preds >= 0.5)
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        return objective
    
    def make_noisy_stack_objective(seed, noise_rate, sample_weights=None):
        def objective(trial):
            try:
                oof_preds, _ = oof_probs_noisy(
                    lambda: build_noisy_stack(trial, seed, noise_rate),
                    X_full, y_full, X_full[:1], noise_rate, sample_weights
                )
                
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_full, oof_preds >= 0.5)
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        return objective
    
    return {
        'stack': make_stack_objective,
        'stack_c': make_stack_c_objective,
        'sklearn': make_sklearn_stack_objective,
        'neural': make_neural_stack_objective,
        'noisy': make_noisy_stack_objective
    }


def main():
    """Main execution function"""
    logger.info("🎯 Six-Stack Personality Classification Pipeline (Modular Version)")
    logger.info("=" * 60)

    # Load data using TOP-4 solution merge strategy
    df_tr, df_te, submission = load_data_with_external_merge()

    # Preprocess data with TOP-4 solution approach
    X_full, X_test, y_full, le = prep(df_tr, df_te)

    # Apply data augmentation after preprocessing
    X_full, y_full = apply_data_augmentation(X_full, y_full)

    # Set up pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    # Create objective functions
    objectives = create_objective_functions(X_full, y_full)

    # Train 6 stacks
    logger.info("\n🔍 Training 6 specialized stacks...")

    # Initialize studies dictionary
    studies = {}

    # Stack A - Traditional ML (narrow hyperparameters)
    logger.info("Training Stack A...")
    studies['A'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    best_params_A = load_best_trial_params("stack_A")
    if best_params_A:
        studies['A'].enqueue_trial(best_params_A)
    studies['A'].optimize(
        objectives['stack'](seed=RND, wide_hp=False, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['A'], "stack_A")

    # Stack B - Traditional ML (wide hyperparameters)
    logger.info("Training Stack B...")
    studies['B'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    best_params_B = load_best_trial_params("stack_B")
    if best_params_B:
        studies['B'].enqueue_trial(best_params_B)
    studies['B'].optimize(
        objectives['stack'](seed=2024, wide_hp=True, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['B'], "stack_B")

    # Stack C - XGBoost + CatBoost
    logger.info("Training Stack C...")
    studies['C'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_C = load_best_trial_params("stack_C")
    if best_params_C:
        studies['C'].enqueue_trial(best_params_C)
    studies['C'].optimize(
        objectives['stack_c'](seed=1337, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['C'], "stack_C")

    # Stack D - Sklearn models
    logger.info("Training Stack D - Sklearn...")
    studies['D'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_D = load_best_trial_params("stack_D")
    if best_params_D:
        studies['D'].enqueue_trial(best_params_D)
    studies['D'].optimize(
        objectives['sklearn'](seed=9999, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['D'], "stack_D")

    # Stack E - Neural networks
    logger.info("Training Stack E - Neural Networks...")
    studies['E'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_E = load_best_trial_params("stack_E")
    if best_params_E:
        studies['E'].enqueue_trial(best_params_E)
    studies['E'].optimize(
        objectives['neural'](seed=7777, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['E'], "stack_E")

    # Stack F - Noisy labels
    logger.info("Training Stack F - Noisy Labels...")
    studies['F'] = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    best_params_F = load_best_trial_params("stack_F")
    if best_params_F:
        studies['F'].enqueue_trial(best_params_F)
    studies['F'].optimize(
        objectives['noisy'](seed=5555, noise_rate=LABEL_NOISE_RATE, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    save_best_trial_params(studies['F'], "stack_F")

    # Create model builders using best parameters
    def builder_A():
        return build_stack(studies['A'].best_trial, seed=RND, wide_hp=False)

    def builder_B():
        return build_stack(studies['B'].best_trial, seed=2024, wide_hp=True)

    def builder_C():
        return build_stack_c(studies['C'].best_trial, seed=1337)

    def builder_D():
        return build_sklearn_stack(studies['D'].best_trial, seed=9999, X_full=X_full)

    def builder_E():
        return build_neural_stack(studies['E'].best_trial, seed=7777, X_full=X_full)

    def builder_F():
        return build_noisy_stack(
            studies['F'].best_trial, seed=5555, noise_rate=LABEL_NOISE_RATE
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
    logger.info("Generating OOF F (noisy)...")
    oof_F, _ = oof_probs_noisy(
        builder_F,
        X_full,
        y_full,
        X_test[:1],
        noise_rate=LABEL_NOISE_RATE,
        sample_weights=None,
    )

    # Optimize blending weights
    logger.info("\n⚖️ Optimizing blending weights...")
    study_blend = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=RND),
    )

    # Enqueue the best performing blend weights
    best_blend_params = {
        "wA": 0.216,
        "wB": 0.093,
        "wC": 0.154,
        "wD": 0.231,
        "wE": 0.152,
        "wF": 0.154,
    }
    study_blend.enqueue_trial(best_blend_params)

    def blend_objective(trial):
        return improved_blend_obj(
            trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_full
        )

    study_blend.optimize(
        blend_objective, n_trials=N_TRIALS_BLEND, show_progress_bar=True
    )

    # Extract best weights
    best_trial = study_blend.best_trial
    best_weights = best_trial.user_attrs["weights"]
    wA, wB, wC, wD, wE, wF = best_weights

    logger.info(
        f"\n🏆 Best blend weights: wA={wA:.3f}, wB={wB:.3f}, wC={wC:.3f}, wD={wD:.3f}, wE={wE:.3f}, wF={wF:.3f}"
    )
    logger.info(f"Best CV score: {study_blend.best_value:.6f}")

    # Refit all models on full data
    logger.info("\n🔄 Refitting all models on full data...")
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

    logger.info("Refitting Stack F (noisy)...")
    mdl_F = builder_F()
    y_full_noisy = add_label_noise(
        y_full, noise_rate=LABEL_NOISE_RATE, random_state=RND
    )
    mdl_F.fit(X_full, y_full_noisy)

    # Generate test predictions for pseudo-labeling
    logger.info("\n🔮 Applying Conservative Pseudo-Labeling...")
    logger.info("Generating test predictions for pseudo-labeling...")

    test_proba_A = mdl_A.predict_proba(X_test)[:, 1]
    test_proba_B = mdl_B.predict_proba(X_test)[:, 1]
    test_proba_C = mdl_C.predict_proba(X_test)[:, 1]
    test_proba_D = mdl_D.predict_proba(X_test)[:, 1]
    test_proba_E = mdl_E.predict_proba(X_test)[:, 1]
    test_proba_F = mdl_F.predict_proba(X_test)[:, 1]

    # Add pseudo-labels with conservative settings
    X_combined, y_combined, pseudo_stats = add_pseudo_labeling_conservative(
        X_full,
        y_full,
        X_test,
        test_proba_A,
        test_proba_B,
        test_proba_C,
        test_proba_D,
        test_proba_E,
        test_proba_F,
        wA,
        wB,
        wC,
        wD,
        wE,
        wF,
        confidence_threshold=0.95,  # Very conservative
        max_pseudo_ratio=0.3,  # Max 30% additional data
    )

    # If pseudo-labels were added, retrain models with combined data
    if pseudo_stats["n_pseudo_added"] > 0:
        logger.info(
            f"\n🔄 Retraining models with {pseudo_stats['n_pseudo_added']} pseudo-labels..."
        )

        logger.info("Retraining Stack A with pseudo-labels...")
        mdl_A_final = builder_A()
        mdl_A_final.fit(X_combined, y_combined)

        logger.info("Retraining Stack B with pseudo-labels...")
        mdl_B_final = builder_B()
        mdl_B_final.fit(X_combined, y_combined)

        logger.info("Retraining Stack C with pseudo-labels...")
        mdl_C_final = builder_C()
        mdl_C_final.fit(X_combined, y_combined)

        logger.info("Retraining Stack D with pseudo-labels...")
        mdl_D_final = builder_D()
        mdl_D_final.fit(X_combined, y_combined)

        logger.info("Retraining Stack E with pseudo-labels...")
        mdl_E_final = builder_E()
        mdl_E_final.fit(X_combined, y_combined)

        logger.info("Retraining Stack F with pseudo-labels...")
        mdl_F_final = builder_F()
        y_combined_noisy = add_label_noise(
            y_combined, noise_rate=LABEL_NOISE_RATE, random_state=RND
        )
        mdl_F_final.fit(X_combined, y_combined_noisy)

        # Use retrained models for final predictions
        logger.info(
            "✅ Using retrained models with pseudo-labels for final predictions"
        )
        final_models = (
            mdl_A_final,
            mdl_B_final,
            mdl_C_final,
            mdl_D_final,
            mdl_E_final,
            mdl_F_final,
        )
    else:
        logger.info("⚠️ No pseudo-labels added, using original models")
        final_models = (mdl_A, mdl_B, mdl_C, mdl_D, mdl_E, mdl_F)

    # Generate final predictions
    logger.info("\n🎯 Generating final predictions...")
    proba_test_continuous = (
        wA * final_models[0].predict_proba(X_test)[:, 1]
        + wB * final_models[1].predict_proba(X_test)[:, 1]
        + wC * final_models[2].predict_proba(X_test)[:, 1]
        + wD * final_models[3].predict_proba(X_test)[:, 1]
        + wE * final_models[4].predict_proba(X_test)[:, 1]
        + wF * final_models[5].predict_proba(X_test)[:, 1]
    )

    # Convert to discrete predictions
    proba_test_discrete = (proba_test_continuous >= 0.5).astype(int)
    personality = le.inverse_transform(proba_test_discrete)

    # Create submission
    import pandas as pd
    submission_df = pd.DataFrame({"id": submission.id, "Personality": personality})

    # Save results
    output_file = "six_stack_personality_predictions_modular.csv"
    submission_df.to_csv(output_file, index=False)

    logger.info(f"\n✅ Predictions saved to '{output_file}'")
    logger.info(f"📊 Final submission shape: {submission_df.shape}")
    logger.info("🎉 Six-stack ensemble pipeline completed successfully!")

    # Print summary
    logger.info("\n📋 Summary:")
    logger.info(f"   - Combined training data: {len(df_tr):,} samples")
    logger.info("   - External data merged as features using TOP-4 solution approach")
    logger.info("   - 6 specialized stacks trained")
    logger.info(f"   - Best ensemble CV score: {study_blend.best_value:.6f}")
    if pseudo_stats["n_pseudo_added"] > 0:
        logger.info(
            f"   - Pseudo-labeling: Added {pseudo_stats['n_pseudo_added']:,} high-confidence samples"
        )
        logger.info(f"   - Final training size: {pseudo_stats['final_size']:,} samples")
        logger.info(
            f"   - Mean pseudo-label confidence: {pseudo_stats['mean_confidence']:.4f}"
        )
    else:
        logger.info("   - Pseudo-labeling: No high-confidence samples found")
    logger.info("   - CPU-only configuration used")
    logger.info("   - Modular architecture for maintainability")


if __name__ == "__main__":
    main()
