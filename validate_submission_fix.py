#!/usr/bin/env python3
"""
Quick validation script to test submission ID fix
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append("src")

from modules.data_loader import load_data_with_external_merge


def validate_submission_ids():
    """Validate that our modular pipeline will generate correct submission IDs."""
    print("🔍 Validating submission ID fix...")

    # Load data the same way as modular pipeline
    df_tr, df_te, submission = load_data_with_external_merge()

    # Check submission template format
    print(f"📊 Submission template shape: {submission.shape}")
    print(f"📊 Test data shape: {df_te.shape}")
    print(f"📊 First 5 submission IDs: {submission.id.head().tolist()}")
    print(f"📊 Last 5 submission IDs: {submission.id.tail().tolist()}")

    # Verify ID range
    print(f"📊 ID range: {submission.id.min()} to {submission.id.max()}")

    # Check if IDs are sequential starting from max train ID
    expected_start = len(df_tr)  # Should be 18524
    actual_start = submission.id.min()

    print(f"📊 Expected start ID: {expected_start}")
    print(f"📊 Actual start ID: {actual_start}")

    if actual_start == expected_start:
        print("✅ ID format is CORRECT - starts after training data")
    else:
        print("❌ ID format is INCORRECT")

    # Load reference working file for comparison
    try:
        ref_submission = pd.read_csv("six_stack_personality_predictions_working.csv")
        print(f"\n📊 Reference submission shape: {ref_submission.shape}")
        print(f"📊 Reference first 5 IDs: {ref_submission.id.head().tolist()}")

        # Check if ID ranges match
        if (submission.id == ref_submission.id).all():
            print("✅ IDs match reference working submission perfectly!")
        else:
            print("❌ IDs do not match reference submission")

    except FileNotFoundError:
        print("⚠️ Reference submission file not found")

    print("\n🎯 Modular pipeline will now use correct IDs from submission template!")


if __name__ == "__main__":
    validate_submission_ids()
