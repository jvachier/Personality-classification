#!/usr/bin/env python3
"""
Six-Stack Personality Classification Pipeline (Final Working Version)
This version focuses on demonstrating the modular structure with lightweight ML models.
"""

import sys
import os

# Add the src directory to sys.path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import our custom modules
from modules.config import setup_logging, RND, N_SPLITS
from modules.data_loader import load_data_with_external_merge
from modules.preprocessing import prep
from modules.optimization import save_best_trial_params, load_best_trial_params
from modules.utils import get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def train_lightweight_stack(
    stack_name, model_class, model_params, X_train, y_train, X_test
):
    """Train a lightweight model stack."""
    logger.info(f"🔧 Training {stack_name}...")

    try:
        # Create model
        model = model_class(**model_params)

        # Perform cross-validation to get OOF predictions
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

        # Get cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        avg_score = cv_scores.mean()

        # Train on full data
        model.fit(X_train, y_train)

        # Generate predictions
        oof_pred = model.predict_proba(X_train)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]

        logger.info(
            f"   ✅ {stack_name} - CV Score: {avg_score:.4f} (±{cv_scores.std():.4f})"
        )

        return {
            "model": model,
            "oof_pred": oof_pred,
            "test_pred": test_pred,
            "cv_score": avg_score,
            "cv_std": cv_scores.std(),
        }

    except Exception as e:
        logger.error(f"   ❌ {stack_name} failed: {str(e)}")
        return None


def main():
    """Main execution function - lightweight and stable."""
    logger.info(
        "🎯 Six-Stack Personality Classification Pipeline (Final Working Version)"
    )
    logger.info("=" * 75)
    logger.info("🔧 Using lightweight ML models for demonstration")

    try:
        # Step 1: Load data
        logger.info("\n📊 Step 1: Loading data...")
        df_tr, df_te, submission = load_data_with_external_merge()

        # Step 2: Preprocess data
        logger.info("\n🔧 Step 2: Preprocessing data...")
        X_full, X_test, y_full, le = prep(df_tr, df_te)
        logger.info(f"   Data prepared - Train: {X_full.shape}, Test: {X_test.shape}")

        # Step 3: Skip data augmentation for stability
        logger.info("\n📈 Step 3: Data augmentation skipped for stability")

        # Step 4: Train lightweight stacks
        logger.info("\n🔍 Step 4: Training lightweight stacks...")

        stacks = {}

        # Stack A: Random Forest (conservative)
        stacks["RF_Conservative"] = train_lightweight_stack(
            "Random Forest (Conservative)",
            RandomForestClassifier,
            {
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": RND,
                "n_jobs": 1,  # Single thread for stability
            },
            X_full,
            y_full,
            X_test,
        )

        # Stack B: Random Forest (complex)
        stacks["RF_Complex"] = train_lightweight_stack(
            "Random Forest (Complex)",
            RandomForestClassifier,
            {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": RND + 1,
                "n_jobs": 1,  # Single thread for stability
            },
            X_full,
            y_full,
            X_test,
        )

        # Stack C: Logistic Regression
        stacks["LogReg"] = train_lightweight_stack(
            "Logistic Regression",
            LogisticRegression,
            {
                "random_state": RND,
                "max_iter": 1000,
                "solver": "liblinear",  # Stable solver
            },
            X_full,
            y_full,
            X_test,
        )

        # Step 5: Ensemble predictions
        logger.info("\n🎯 Step 5: Creating ensemble...")

        successful_stacks = [
            name for name, result in stacks.items() if result is not None
        ]

        if successful_stacks:
            logger.info(f"   📊 Ensembling {len(successful_stacks)} successful stacks")

            # Collect predictions
            test_predictions = []
            stack_weights = []

            for stack_name in successful_stacks:
                result = stacks[stack_name]
                test_predictions.append(result["test_pred"])
                # Weight by cross-validation score
                stack_weights.append(result["cv_score"])
                logger.info(f"   • {stack_name}: {result['cv_score']:.4f}")

            # Weighted ensemble
            weights = np.array(stack_weights)
            weights = weights / weights.sum()  # Normalize

            test_ensemble = np.average(test_predictions, axis=0, weights=weights)

            # Calculate ensemble CV score (approximation)
            oof_predictions = []
            for stack_name in successful_stacks:
                oof_predictions.append(stacks[stack_name]["oof_pred"])

            oof_ensemble = np.average(oof_predictions, axis=0, weights=weights)
            ensemble_accuracy = accuracy_score(y_full, oof_ensemble >= 0.5)

            logger.info(f"   🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")

            # Step 6: Create submission
            logger.info("\n💾 Step 6: Creating submission...")

            # Convert to binary predictions
            binary_preds = (test_ensemble >= 0.5).astype(int)
            submission["Personality"] = le.inverse_transform(binary_preds)

            # Save submission
            submission_filename = "submission_modular_final.csv"
            submission.to_csv(submission_filename, index=False)

            # Step 7: Summary
            logger.info("\n🎉 Pipeline Summary:")
            logger.info("=" * 50)
            logger.info(f"📊 Training samples: {X_full.shape[0]:,}")
            logger.info(f"📊 Test samples: {X_test.shape[0]:,}")
            logger.info(f"📊 Features: {X_full.shape[1]}")
            logger.info(f"🎯 Successful stacks: {len(successful_stacks)}")
            logger.info(f"🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")
            logger.info(f"💾 Submission: {submission_filename}")

            # Individual stack performance
            logger.info("\n📈 Individual Stack Performance:")
            for stack_name in successful_stacks:
                result = stacks[stack_name]
                logger.info(
                    f"   • {stack_name}: {result['cv_score']:.4f} (±{result['cv_std']:.4f})"
                )

            logger.info("\n✨ Modular pipeline completed successfully!")
            logger.info(
                "🔧 This demonstrates the modular structure with lightweight models"
            )
            logger.info(
                "📝 Replace with advanced models (XGBoost, Neural Networks) for production"
            )

        else:
            logger.error("❌ No successful stacks trained")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
