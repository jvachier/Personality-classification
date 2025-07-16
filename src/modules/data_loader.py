"""Data loading functions for the personality classification pipeline."""

import pandas as pd

from .config import Paths
from .utils import get_logger

logger = get_logger(__name__)


def load_data_with_external_merge():
    """Load and merge training data with external personality datasets using advanced merge strategy.

    This function merges external data as features rather than concatenating as new samples.

    Returns:
        tuple: (df_tr, df_te, submission) - training data, test data, and submission template
    """
    logger.info("📊 Loading data with advanced merge strategy...")

    # Use Paths enum from config.py for all file paths
    df_tr = pd.read_csv(Paths.TRAIN_CSV.value)
    df_te = pd.read_csv(Paths.TEST_CSV.value)
    submission = pd.read_csv(Paths.SAMPLE_SUBMISSION_CSV.value)

    logger.info(f"Original train shape: {df_tr.shape}")
    logger.info(f"Original test shape: {df_te.shape}")

    # Load external dataset using advanced merge strategy

    try:
        df_external = pd.read_csv(Paths.PERSONALITY_DATASET_CSV.value)
        logger.info(f"External dataset shape: {df_external.shape}")

        # Rename Personality column to match_p for clarity
        df_external = df_external.rename(columns={"Personality": "match_p"})

        # Define merge columns (all the features except target and id)
        merge_cols = [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]

        # Remove duplicates based on feature combinations to avoid conflicts
        original_external_shape = df_external.shape[0]
        df_external = df_external.drop_duplicates(subset=merge_cols)
        duplicates_removed = original_external_shape - df_external.shape[0]
        logger.info(f"Removed {duplicates_removed} duplicate rows from external data")
        logger.info(f"External dataset shape after deduplication: {df_external.shape}")

        # Merge with training and test data to create match_p feature
        df_tr = df_tr.merge(df_external, how="left", on=merge_cols)
        df_te = df_te.merge(df_external, how="left", on=merge_cols)

        # Count successful matches
        train_matches = df_tr["match_p"].notna().sum()
        test_matches = df_te["match_p"].notna().sum()

        logger.info(
            f"✅ Successfully matched {train_matches}/{len(df_tr)} training samples with external data"
        )
        logger.info(
            f"✅ Successfully matched {test_matches}/{len(df_te)} test samples with external data"
        )

        # Print match distribution for training data
        if train_matches > 0:
            match_dist = df_tr["match_p"].value_counts(dropna=False)
            logger.info("Training match_p distribution:")
            for value, count in match_dist.items():
                logger.info(f"   {value}: {count} ({count / len(df_tr) * 100:.1f}%)")

    except FileNotFoundError:
        logger.warning(
            "⚠️ personality_datasert.csv not found, adding empty match_p column"
        )
        df_tr["match_p"] = None
        df_te["match_p"] = None

    logger.info("✅ Data loading with external merge completed")
    return df_tr, df_te, submission
