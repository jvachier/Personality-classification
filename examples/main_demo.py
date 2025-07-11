#!/usr/bin/env python3
"""
Six-Stack Personality Classification Pipeline (Minimal Working Demo)
This version demonstrates the modular structure without heavy ML operations.
"""

import sys
import os

# Add the src directory to sys.path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Import our custom modules
from modules.config import setup_logging, RND, ENABLE_DATA_AUGMENTATION
from modules.data_loader import load_data_with_external_merge
from modules.preprocessing import prep
from modules.utils import get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def create_dummy_predictions(X_train, y_train, X_test, stack_name):
    """Create dummy predictions for demonstration."""
    logger.info(f"   🤖 Training {stack_name} (dummy model for demo)...")

    # Use a simple dummy classifier for demonstration
    model = DummyClassifier(strategy="stratified", random_state=RND)
    model.fit(X_train, y_train)

    # Generate predictions
    oof_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_train, oof_pred >= 0.5)
    logger.info(f"   ✅ {stack_name} completed - Accuracy: {accuracy:.4f}")

    return oof_pred, test_pred, accuracy


def main():
    """Main execution function - minimal working demo."""
    logger.info("🎯 Six-Stack Personality Classification Pipeline (Demo Version)")
    logger.info("=" * 70)
    logger.info("🔍 This is a demonstration of the modular structure with dummy models")

    try:
        # Load data
        logger.info("\n📊 Step 1: Loading data...")
        df_tr, df_te, submission = load_data_with_external_merge()

        # Preprocess data
        logger.info("\n🔧 Step 2: Preprocessing data...")
        X_full, X_test, y_full, le = prep(df_tr, df_te)

        logger.info(
            f"   ✅ Data prepared - Train: {X_full.shape}, Test: {X_test.shape}"
        )

        # Optional data augmentation (skip for demo to avoid hanging)
        if ENABLE_DATA_AUGMENTATION:
            logger.info("\n📈 Step 3: Data augmentation...")
            logger.info("   ⏭️ Skipping data augmentation in demo mode for stability")
        else:
            logger.info("\n📈 Step 3: Data augmentation disabled")

        # Train multiple stacks (using dummy models for demo)
        logger.info("\n🔍 Step 4: Training demonstration stacks...")

        stack_results = {}
        stack_names = ["Stack_A", "Stack_B", "Stack_C"]

        all_oof_preds = []
        all_test_preds = []

        for stack_name in stack_names:
            oof_pred, test_pred, accuracy = create_dummy_predictions(
                X_full, y_full, X_test, stack_name
            )

            stack_results[stack_name] = {
                "oof_pred": oof_pred,
                "test_pred": test_pred,
                "accuracy": accuracy,
            }

            all_oof_preds.append(oof_pred)
            all_test_preds.append(test_pred)

        # Ensemble predictions (simple averaging)
        logger.info("\n📈 Step 5: Ensemble blending...")

        oof_ensemble = np.mean(all_oof_preds, axis=0)
        test_ensemble = np.mean(all_test_preds, axis=0)

        # Calculate ensemble accuracy
        ensemble_accuracy = accuracy_score(y_full, oof_ensemble >= 0.5)
        logger.info(f"   🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")

        # Create submission file
        logger.info("\n💾 Step 6: Creating submission...")

        # Convert predictions to binary
        binary_test_preds = (test_ensemble >= 0.5).astype(int)

        # Update submission dataframe
        submission["Personality"] = le.inverse_transform(binary_test_preds)

        # Save submission
        submission_filename = "submission_modular_demo.csv"
        submission.to_csv(submission_filename, index=False)

        logger.info(f"   ✅ Submission saved to {submission_filename}")

        # Summary
        logger.info("\n🎉 Pipeline Summary:")
        logger.info("=" * 50)
        logger.info(f"📊 Training samples: {X_full.shape[0]:,}")
        logger.info(f"📊 Test samples: {X_test.shape[0]:,}")
        logger.info(f"📊 Features: {X_full.shape[1]}")
        logger.info(f"🎯 Stacks trained: {len(stack_names)}")
        logger.info(f"🎯 Final ensemble accuracy: {ensemble_accuracy:.4f}")

        for stack_name, results in stack_results.items():
            logger.info(f"   • {stack_name}: {results['accuracy']:.4f}")

        logger.info(f"💾 Submission file: {submission_filename}")
        logger.info("\n✨ Modular pipeline demo completed successfully!")
        logger.info(
            "📝 Note: This demo uses dummy models. Replace with real ML models for production."
        )

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
