#!/usr/bin/env python3
"""
Six-Stack Personality Classification Script
Converted from Jupyter notebook with CPU-only configuration and external data integration
"""

import gc
import warnings
import sys
import logging
from typing import Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    RobustScaler,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import optuna
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("personality_classifier.log"),
    ],
)
logger = logging.getLogger(__name__)

# Data augmentation imports
try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    logger.warning("⚠️ SDV not available. Install with: pip install sdv")

try:
    from imblearn.over_sampling import SMOTENC

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning(
        "⚠️ imbalanced-learn not available. Install with: pip install imbalanced-learn"
    )

warnings.filterwarnings("ignore")
# Suppress joblib resource tracker warnings on macOS
warnings.filterwarnings("ignore", message="resource_tracker")

# Global parameters
RND = 42
N_SPLITS = 5

# Data augmentation parameters
ENABLE_DATA_AUGMENTATION = True  # Set to True to enable
AUGMENTATION_METHOD = (
    "sdv_copula"  # Options: "simple", "sdv_copula", "sdv_ctgan", "smotenc"
    # Note: Use "sdv_copula" on Apple Silicon - CTGAN may hang on M1/M2/M3
)
AUGMENTATION_RATIO = 0.05  # 5% additional synthetic data
QUALITY_THRESHOLD = 0.8  # For quality filtering
N_TRIALS_STACK = 15
N_TRIALS_BLEND = 200
LABEL_NOISE_RATE = 0.02


def load_data_with_external_merge():
    """
    Load and merge training data with external personality datasets using TOP-4 solution strategy.
    This function merges external data as features rather than concatenating as new samples.
    """
    logger.info("📊 Loading data with TOP-4 solution merge strategy...")

    # Load original datasets
    df_tr = pd.read_csv("./data/train.csv")
    df_te = pd.read_csv("./data/test.csv")
    submission = pd.read_csv("./data/sample_submission.csv")

    logger.info(f"Original train shape: {df_tr.shape}")
    logger.info(f"Original test shape: {df_te.shape}")

    # Load external dataset using TOP-4 solution merge strategy
    try:
        df_external = pd.read_csv("./data/personality_datasert.csv")
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
        # This adds the match_p column as a new feature
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


def simple_mixed_augmentation(X_train, y_train, augment_ratio=0.05):
    """Simple data augmentation for mixed numerical/categorical features"""

    # Define feature types based on your dataset
    numerical_features = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    categorical_features = ["Stage_fear", "Drained_after_socializing"]

    # Balance classes - augment minority class
    class_counts = y_train.value_counts()
    minority_class = class_counts.idxmin()
    minority_mask = y_train == minority_class
    minority_X = X_train[minority_mask]

    if len(minority_X) == 0:
        return pd.DataFrame(), pd.Series()

    n_augment = int(len(minority_X) * augment_ratio)
    if n_augment == 0:
        return pd.DataFrame(), pd.Series()

    augment_indices = np.random.choice(len(minority_X), n_augment, replace=True)
    base_samples = minority_X.iloc[augment_indices].copy()

    # Augment numerical features with noise
    for col in numerical_features:
        if col in base_samples.columns and not base_samples[col].isna().all():
            # Add gaussian noise (5% of feature std)
            feature_std = X_train[col].std()
            if feature_std > 0:
                noise_std = feature_std * 0.05
                noise = np.random.normal(0, noise_std, len(base_samples))
                base_samples[col] = base_samples[col] + noise

                # Clip to valid range
                col_min, col_max = X_train[col].min(), X_train[col].max()
                base_samples[col] = np.clip(base_samples[col], col_min, col_max)

    # Augment categorical features with probability flipping
    for col in categorical_features:
        if col in base_samples.columns:
            # 5% chance to flip categorical values
            flip_mask = np.random.random(len(base_samples)) < 0.05
            unique_vals = X_train[col].dropna().unique()

            for idx in base_samples[flip_mask].index:
                current_val = base_samples.loc[idx, col]
                if pd.notna(current_val) and len(unique_vals) > 1:
                    # Choose different value
                    other_vals = [v for v in unique_vals if v != current_val]
                    if other_vals:
                        base_samples.loc[idx, col] = np.random.choice(other_vals)

    # Create labels
    augmented_y = pd.Series([minority_class] * n_augment, index=base_samples.index)

    return base_samples, augmented_y


def sdv_augmentation(X_train, y_train, method="copula", augment_ratio=0.05):
    """
    High-quality synthetic data generation using SDV with improved CTGAN handling
    """
    if not SDV_AVAILABLE:
        logger.warning("⚠️ SDV not available, falling back to simple augmentation")
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)

    try:
        logger.info(f"   🔧 Using {method} synthesizer...")

        # Combine features and target for SDV
        train_data = X_train.copy()
        train_data["Personality"] = y_train

        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        # Update metadata for categorical columns (handle one-hot encoded features)
        categorical_cols = []
        for col in train_data.columns:
            if col.startswith(
                ("Stage_fear_", "Drained_after_socializing_", "match_p_")
            ):
                categorical_cols.append(col)
                metadata.update_column(col, sdtype="categorical")

        metadata.update_column("Personality", sdtype="categorical")

        # Choose synthesizer based on method
        if method == "copula":
            logger.info("   ⚡ Training GaussianCopula (fast)...")
            synthesizer = GaussianCopulaSynthesizer(metadata)
        elif method == "ctgan":
            # Check for Apple Silicon - CTGAN has issues on M1/M2/M3
            import platform

            is_apple_silicon = (
                platform.machine() == "arm64" and platform.system() == "Darwin"
            )

            if is_apple_silicon:
                logger.warning(
                    "   ⚠️ Apple Silicon detected - CTGAN has compatibility issues"
                )
                logger.info("   🔄 Falling back to GaussianCopula for stability...")
                synthesizer = GaussianCopulaSynthesizer(metadata)
            else:
                logger.info("   🧠 Training CTGAN (slow but high quality)...")

                # Much more conservative CTGAN parameters to prevent hanging
                data_size = len(train_data)
                if data_size > 10000:
                    epochs = 5  # Even fewer epochs for large datasets
                    batch_size = 500  # Smaller batch size
                elif data_size > 5000:
                    epochs = 10
                    batch_size = 250
                else:
                    epochs = 20
                    batch_size = min(100, data_size // 8)

                logger.info(f"   📊 Using epochs={epochs}, batch_size={batch_size}")

                try:
                    synthesizer = CTGANSynthesizer(
                        metadata,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=False,  # Disable verbose to prevent hanging
                        cuda=False,  # Ensure CPU-only
                        discriminator_decay=1e-5,
                        generator_decay=1e-5,
                        discriminator_steps=1,
                        pac=1,
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ CTGAN initialization failed: {e}")
                    logger.info("   🔄 Falling back to GaussianCopula...")
                    synthesizer = GaussianCopulaSynthesizer(metadata)
        else:
            logger.info("   ⚡ Fallback to GaussianCopula...")
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer = GaussianCopulaSynthesizer(metadata)

        # Fit the synthesizer with timeout protection
        logger.info("   🏋️ Training synthesizer...")

        import signal
        import time

        def timeout_handler(signum, frame):
            raise TimeoutError("Synthesizer training timed out")

        start_time = time.time()

        # Set timeout for CTGAN (3 minutes max)
        if method == "ctgan":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)  # 3 minutes timeout

        try:
            synthesizer.fit(train_data)
            training_time = time.time() - start_time
            logger.info(f"   ✅ Training completed in {training_time:.1f}s")
        finally:
            if method == "ctgan":
                signal.alarm(0)  # Cancel timeout

        # Generate synthetic data
        n_synthetic = int(len(train_data) * augment_ratio)
        if n_synthetic == 0:
            return pd.DataFrame(), pd.Series()

        logger.info(f"   🎲 Generating {n_synthetic} synthetic samples...")
        synthetic_data = synthesizer.sample(num_rows=n_synthetic)

        # Split back to features and target
        synthetic_X = synthetic_data.drop("Personality", axis=1)
        synthetic_y = synthetic_data["Personality"]

        logger.info(f"   ✅ Generated {len(synthetic_X)} synthetic samples")
        return synthetic_X, synthetic_y

    except TimeoutError:
        logger.warning(
            "⚠️ CTGAN training timed out (3 min), falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)
    except Exception as e:
        logger.warning(
            f"⚠️ SDV augmentation failed: {str(e)}, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)


def smotenc_augmentation(X_train, y_train):
    """SMOTE for mixed numerical/categorical data"""
    if not IMBLEARN_AVAILABLE:
        logger.warning(
            "⚠️ imbalanced-learn not available, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, 0.1)

    try:
        # Encode categorical features for SMOTE
        X_encoded = X_train.copy()

        # Define categorical features
        categorical_features = ["Stage_fear", "Drained_after_socializing"]
        categorical_indices = []

        label_encoders = {}
        for i, col in enumerate(X_train.columns):
            if col in categorical_features:
                le = LabelEncoder()
                # Handle NaN values
                X_encoded[col] = X_encoded[col].fillna("Unknown")
                X_encoded[col] = le.fit_transform(X_encoded[col])
                label_encoders[col] = le
                categorical_indices.append(i)

        # Apply SMOTENC
        smote = SMOTENC(categorical_features=categorical_indices, random_state=RND)
        X_resampled, y_resampled = smote.fit_resample(X_encoded, y_train)

        # Decode categorical features back
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        for col in categorical_features:
            if col in label_encoders:
                X_resampled[col] = label_encoders[col].inverse_transform(
                    X_resampled[col].astype(int)
                )

        # Return only the augmented samples (not the original ones)
        n_original = len(X_train)
        augmented_X = X_resampled.iloc[n_original:].copy()
        augmented_y = pd.Series(y_resampled[n_original:])

        return augmented_X, augmented_y

    except Exception as e:
        logger.warning(
            f"⚠️ SMOTENC augmentation failed: {str(e)}, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, 0.1)


def apply_data_augmentation(X_train, y_train):
    """Apply the configured data augmentation method"""
    if not ENABLE_DATA_AUGMENTATION:
        logger.info("📊 Data augmentation disabled")
        return X_train, y_train

    logger.info(f"📊 Applying {AUGMENTATION_METHOD} data augmentation...")
    original_shape = X_train.shape

    if AUGMENTATION_METHOD == "simple":
        augmented_X, augmented_y = simple_mixed_augmentation(
            X_train, y_train, AUGMENTATION_RATIO
        )
    elif AUGMENTATION_METHOD == "sdv_copula":
        augmented_X, augmented_y = sdv_augmentation(
            X_train, y_train, "copula", AUGMENTATION_RATIO
        )
    elif AUGMENTATION_METHOD == "sdv_ctgan":
        augmented_X, augmented_y = sdv_augmentation(
            X_train, y_train, "ctgan", AUGMENTATION_RATIO
        )
    elif AUGMENTATION_METHOD == "smotenc":
        augmented_X, augmented_y = smotenc_augmentation(X_train, y_train)
    else:
        logger.warning(f"⚠️ Unknown augmentation method: {AUGMENTATION_METHOD}")
        return X_train, y_train

    if len(augmented_X) > 0:
        # Combine original and augmented data
        X_combined = pd.concat([X_train, augmented_X], ignore_index=True)
        y_combined = pd.concat([y_train, augmented_y], ignore_index=True)

        logger.info(f"   ✅ Added {len(augmented_X)} synthetic samples")
        logger.info(f"   📈 Data shape: {original_shape} → {X_combined.shape}")

        return X_combined, y_combined
    else:
        logger.warning("   ⚠️ No synthetic samples generated")
        return X_train, y_train


def augment_data_conservative(
    df_tr: pd.DataFrame,
    augment_ratio: float = 0.05,  # Very conservative ratio
    random_state: int = 42,
) -> pd.DataFrame:
    """
    DEPRECATED: Use apply_data_augmentation instead.
    Conservative data augmentation using noise injection and feature perturbation.
    This is safer than synthetic generation and less likely to hurt performance.
    """
    logger.info(
        f"🔄 Conservative augmentation with noise injection (ratio: {augment_ratio:.1%})..."
    )

    try:
        # Number of samples to generate
        n_synthetic = int(len(df_tr) * augment_ratio)
        logger.info(f"   🎲 Generating {n_synthetic:,} augmented samples...")

        # Features to augment (exclude target and categorical)
        numerical_cols = [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]

        # Sample from original data
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(df_tr), size=n_synthetic, replace=True)
        augmented_data = df_tr.iloc[sample_indices].copy().reset_index(drop=True)

        # Add small amounts of noise to numerical features
        for col in numerical_cols:
            if col in augmented_data.columns:
                # Calculate noise level as small fraction of feature std
                noise_std = augmented_data[col].std() * 0.05  # 5% of std
                noise = np.random.normal(0, noise_std, size=len(augmented_data))

                # Add noise and clip to original range
                col_min, col_max = df_tr[col].min(), df_tr[col].max()
                augmented_data[col] = np.clip(
                    augmented_data[col] + noise, col_min, col_max
                )

        # Combine with original data
        df_tr_copy = df_tr.copy()
        df_tr_copy["is_synthetic"] = 0
        augmented_data["is_synthetic"] = 1

        df_combined = pd.concat([df_tr_copy, augmented_data], ignore_index=True)

        logger.info("   ✅ Conservative augmentation completed!")
        logger.info(f"   📊 Original samples: {len(df_tr):,}")
        logger.info(f"   📊 Augmented samples: {len(augmented_data):,}")
        logger.info(f"   📊 Total samples: {len(df_combined):,}")

        return df_combined

    except Exception as e:
        logger.warning(f"   ⚠️ Conservative augmentation failed: {str(e)}")
        logger.info("   🔄 Continuing with original data...")
        df_tr_copy = df_tr.copy()
        df_tr_copy["is_synthetic"] = 0
        return df_tr_copy


def prep(
    df_tr: pd.DataFrame, df_te: pd.DataFrame, tgt="Personality", idx="id"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the training and test datasets with TOP-4 solution approach.
    """
    logger.info("🔧 Preprocessing data with TOP-4 solution approach...")

    # Define feature groups before any processing
    # Keep original column types for proper categorization
    original_columns = df_tr.columns.tolist()
    if tgt in original_columns:
        original_columns.remove(tgt)
    if idx in original_columns:
        original_columns.remove(idx)

    # Downcast numerical columns first
    for col in df_tr.select_dtypes(include=["float64", "int64"]).columns:
        if col not in [tgt, idx]:
            df_tr[col] = pd.to_numeric(df_tr[col], downcast="float")
            if col in df_te.columns:
                df_te[col] = pd.to_numeric(df_te[col], downcast="float")

    # Drop the index column if it exists
    if idx in df_tr.columns:
        df_tr = df_tr.drop(columns=[idx])
    if idx in df_te.columns:
        df_te = df_te.drop(columns=[idx])

    # Use TOP-4 solution correlation-based imputation
    logger.info("🔄 Performing TOP-4 correlation-based imputation...")

    # Extract and encode target variable BEFORE combining data
    le_tgt = LabelEncoder()
    ytr = pd.Series(le_tgt.fit_transform(df_tr[tgt]), name=tgt)

    # Remove target column from training data before combining
    df_tr_no_target = df_tr.drop(columns=[tgt])

    # Combine train and test for imputation (as in TOP-4 solution)
    ntrain = len(df_tr_no_target)
    all_data = pd.concat([df_tr_no_target, df_te], ignore_index=True)

    def fill_missing_by_quantile_group(
        df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0]
    ):
        """Fill missing values using correlation-based grouping (from TOP-4 solution)"""
        if target_col not in df.columns or group_source_col not in df.columns:
            return df

        labels = [f"Q{i + 1}" for i in range(len(quantiles) - 1)]
        temp_bin_col = f"{group_source_col}_bin"

        # Create grouping column
        try:
            df[temp_bin_col] = pd.qcut(
                df[group_source_col], q=quantiles, labels=labels, duplicates="drop"
            )
        except (ValueError, TypeError):
            # If qcut fails, use regular binning
            df[temp_bin_col] = pd.cut(
                df[group_source_col], bins=len(labels), labels=labels
            )

        # Fill missing values within groups using median
        df[target_col] = df[target_col].fillna(
            df.groupby(temp_bin_col)[target_col].transform("median")
        )

        # Drop temporary column
        df.drop(columns=[temp_bin_col], inplace=True)
        return df

    # Sequential imputation based on correlations (from TOP-4 solution)
    # 1. Time_spent_Alone using Social_event_attendance
    if (
        "Social_event_attendance" in all_data.columns
        and "Time_spent_Alone" in all_data.columns
    ):
        all_data = fill_missing_by_quantile_group(
            all_data, "Social_event_attendance", "Time_spent_Alone"
        )

    # 2. Time_spent_Alone using Going_outside (for remaining missing values)
    if "Going_outside" in all_data.columns and "Time_spent_Alone" in all_data.columns:
        all_data = fill_missing_by_quantile_group(
            all_data, "Going_outside", "Time_spent_Alone"
        )

    # 3. Social_event_attendance using Going_outside
    if (
        "Going_outside" in all_data.columns
        and "Social_event_attendance" in all_data.columns
    ):
        all_data = fill_missing_by_quantile_group(
            all_data, "Going_outside", "Social_event_attendance"
        )

    # 4. Fill remaining features using correlated features
    feature_pairs = [
        ("Friends_circle_size", "Social_event_attendance"),
        ("Post_frequency", "Friends_circle_size"),
        ("Going_outside", "Social_event_attendance"),
        ("Friends_circle_size", "Going_outside"),
        ("Post_frequency", "Going_outside"),
    ]

    for target_col, source_col in feature_pairs:
        if target_col in all_data.columns and source_col in all_data.columns:
            all_data = fill_missing_by_quantile_group(all_data, source_col, target_col)

    # Handle categorical features with "Unknown" (from TOP-4 solution)
    categorical_cols = ["Stage_fear", "Drained_after_socializing", "match_p"]
    for col in categorical_cols:
        if col in all_data.columns:
            # Fill missing values with 'Unknown'
            all_data[col] = all_data[col].fillna("Unknown")

    # Apply one-hot encoding for categorical features using OneHotEncoder (TOP-4 solution approach)
    logger.info("🔄 Applying one-hot encoding for categorical features...")

    # Identify categorical columns that exist in the data
    existing_categorical_cols = [
        col for col in categorical_cols if col in all_data.columns
    ]

    if existing_categorical_cols:
        # Use OneHotEncoder for better control and consistency
        encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")

        # Extract categorical data
        categorical_data = all_data[existing_categorical_cols]

        # Fit and transform the categorical data
        encoded_data = encoder.fit_transform(categorical_data)

        # Get feature names for the encoded columns
        feature_names = encoder.get_feature_names_out(existing_categorical_cols)

        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_data, columns=feature_names, index=all_data.index
        )

        # Remove original categorical columns and add encoded ones
        all_data = all_data.drop(columns=existing_categorical_cols)
        all_data = pd.concat([all_data, encoded_df], axis=1)

        logger.info(
            f"   ✅ Encoded {len(existing_categorical_cols)} categorical features into {len(feature_names)} binary features"
        )
    else:
        logger.warning("   ⚠️ No categorical features found to encode")

    # Fill any remaining missing values
    logger.info("🔄 Filling any remaining missing values...")

    # For numerical columns, use median
    num_cols = all_data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if all_data[col].isnull().any():
            all_data[col].fillna(all_data[col].median(), inplace=True)

    # For any remaining categorical columns (shouldn't be any after one-hot encoding)
    cat_cols = all_data.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if all_data[col].isnull().any():
            all_data[col].fillna("Unknown", inplace=True)

    # Split back to train and test
    df_tr = all_data[:ntrain].copy()
    df_te = all_data[ntrain:].copy()

    logger.info(f"Final train shape: {df_tr.shape}")
    logger.info(f"Final test shape: {df_te.shape}")

    logger.info("✅ Preprocessing completed with TOP-4 solution approach")
    return df_tr, df_te, ytr, le_tgt


def add_label_noise(y, noise_rate=0.02, random_state=42):
    """
    Add controlled label noise for regularization.
    """
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_indices = np.random.choice(len(y), n_flip, replace=False)

    # Flip labels (0->1, 1->0)
    y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]

    return y_noisy


def add_pseudo_labeling_conservative(
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
    confidence_threshold=0.95,
    max_pseudo_ratio=0.3,
):
    """
    Add high-confidence pseudo-labels to training data using ensemble predictions.

    Args:
        X_full, y_full: Original training data
        X_test: Test data for pseudo-labeling
        test_proba_*: Individual model predictions on test set
        w*: Ensemble weights from blending optimization
        confidence_threshold: Minimum confidence for pseudo-labels (default 0.95)
        max_pseudo_ratio: Maximum ratio of pseudo-labels to original training data

    Returns:
        Tuple of (X_combined, y_combined, pseudo_stats)
    """
    logger.info(
        "🔮 Adding pseudo-labels with confidence threshold {confidence_threshold}"
    )

    # Generate ensemble predictions on test set using optimized weights
    ensemble_proba = (
        wA * test_proba_A
        + wB * test_proba_B
        + wC * test_proba_C
        + wD * test_proba_D
        + wE * test_proba_E
        + wF * test_proba_F
    )

    # Find high-confidence predictions (very confident in either class)
    high_conf_mask = (ensemble_proba >= confidence_threshold) | (
        ensemble_proba <= (1 - confidence_threshold)
    )

    n_high_conf = np.sum(high_conf_mask)
    max_pseudo_samples = int(len(X_full) * max_pseudo_ratio)

    logger.info("   📊 Found {n_high_conf} high-confidence predictions")
    logger.info("   📏 Maximum allowed pseudo-samples: {max_pseudo_samples}")

    if n_high_conf > 0:
        # Limit pseudo-samples to avoid overfitting
        if n_high_conf > max_pseudo_samples:
            logger.info("   ✂️ Limiting to {max_pseudo_samples} most confident samples")
            # Get confidence scores and select most confident samples
            conf_scores = np.maximum(ensemble_proba, 1 - ensemble_proba)
            high_conf_indices = np.where(high_conf_mask)[0]
            selected_indices = high_conf_indices[
                np.argsort(conf_scores[high_conf_indices])[-max_pseudo_samples:]
            ]
            final_mask = np.zeros_like(high_conf_mask)
            final_mask[selected_indices] = True
        else:
            final_mask = high_conf_mask

        # Get high-confidence test samples
        X_pseudo = X_test[final_mask].copy()
        y_pseudo = (ensemble_proba[final_mask] >= 0.5).astype(int)

        # Combine with original training data
        X_combined = pd.concat([X_full, X_pseudo], axis=0, ignore_index=True)
        y_combined = pd.concat(
            [y_full, pd.Series(y_pseudo, name="Personality")], axis=0, ignore_index=True
        )

        # Statistics
        pseudo_stats = {
            "n_pseudo_added": len(y_pseudo),
            "pseudo_class_0": np.sum(y_pseudo == 0),
            "pseudo_class_1": np.sum(y_pseudo == 1),
            "mean_confidence": np.mean(
                np.maximum(ensemble_proba[final_mask], 1 - ensemble_proba[final_mask])
            ),
            "original_size": len(X_full),
            "final_size": len(X_combined),
        }

        logger.info(f"   ✅ Added {len(y_pseudo)} pseudo-labels to training data")
        logger.info(
            f"   📊 Pseudo-label distribution: Class 0: {pseudo_stats['pseudo_class_0']}, Class 1: {pseudo_stats['pseudo_class_1']}"
        )
        logger.info(f"   🎯 Mean confidence: {pseudo_stats['mean_confidence']:.4f}")
        logger.info(f"   📈 Training data: {len(X_full)} → {len(X_combined)} samples")

        return X_combined, y_combined, pseudo_stats
    else:
        logger.info("   ⚠️ No high-confidence predictions found")
        pseudo_stats = {
            "n_pseudo_added": 0,
            "original_size": len(X_full),
            "final_size": len(X_full),
        }
        return X_full, y_full, pseudo_stats


def create_domain_balanced_dataset(
    dataframes: List[pd.DataFrame],
    target_column: str = "Personality",
    domain_names: List[str] = None,
    random_state: int = 42,
    filter_low_quality: bool = True,
    weight_threshold: float = 0.2,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create domain-balanced dataset using inverse-propensity weighting.
    First dataframe is used as reference distribution.

    Args:
        dataframes: List of dataframes [reference_df, additional_df1, ...]
        target_column: Name of target column to exclude from features
        domain_names: Optional names for each domain
        random_state: Random seed
        filter_low_quality: Whether to filter out low-quality external samples
        weight_threshold: Minimum weight to keep samples (filters very different samples)

    Returns:
        Tuple of (combined_dataframe, sample_weights)
    """
    logger.info("🎯 Computing domain weights for distribution alignment...")

    # Combine dataframes with domain labels
    combined_data = []
    domain_labels = []

    for i, df in enumerate(dataframes):
        combined_data.append(df.copy())
        domain_labels.extend([i] * len(df))

    # Create combined dataset
    combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
    domain_labels = np.array(domain_labels)

    # Identify features (exclude target and id columns)
    feature_cols = [
        col for col in combined_df.columns if col not in [target_column, "id"]
    ]

    # Separate categorical and numerical features
    categorical_features = []
    numerical_features = []

    for col in feature_cols:
        if (
            combined_df[col].dtype == "object"
            or combined_df[col].dtype.name == "category"
        ):
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    # Handle missing values first
    # Fill numerical features with median
    for col in numerical_features:
        if combined_df[col].isnull().any():
            combined_df[col].fillna(combined_df[col].median(), inplace=True)

    # Fill categorical features with mode or 'missing'
    for col in categorical_features:
        if combined_df[col].isnull().any():
            combined_df[col].fillna("missing", inplace=True)

    # Create preprocessing pipeline
    transformers = []
    if numerical_features:
        transformers.append(("num", StandardScaler(), numerical_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit and transform features
    X_processed = preprocessor.fit_transform(combined_df)

    # Train domain classifier
    domain_classifier = LogisticRegression(
        max_iter=1000, random_state=random_state, multi_class="ovr"
    )
    domain_classifier.fit(X_processed, domain_labels)

    # Get propensity scores
    propensity_scores = domain_classifier.predict_proba(X_processed)

    # Calculate weights: reference domain (0) gets weight 1.0, others get reweighted
    weights = np.ones(len(domain_labels))

    # Reference domain samples keep weight = 1.0
    reference_mask = domain_labels == 0
    weights[reference_mask] = 1.0

    # Other domains get inverse-propensity weights to match reference distribution
    for domain_idx in range(1, len(dataframes)):
        domain_mask = domain_labels == domain_idx
        if np.any(domain_mask):
            # Weight = P(reference) / P(current_domain)
            ref_probs = propensity_scores[
                domain_mask, 0
            ]  # Prob of being in reference domain
            domain_probs = propensity_scores[
                domain_mask, domain_idx
            ]  # Prob of being in current domain
            weights[domain_mask] = ref_probs / (domain_probs + 1e-8)

    # Normalize weights per domain to have mean 1.0
    for domain_idx in range(len(dataframes)):
        domain_mask = domain_labels == domain_idx
        if np.any(domain_mask):
            domain_weights = weights[domain_mask]
            weights[domain_mask] = domain_weights / (np.mean(domain_weights) + 1e-8)

    # Print summary
    logger.info("📊 Domain weighting summary:")
    logger.info("   Reference domain: 0 (first dataframe)")
    for domain_idx in range(len(dataframes)):
        domain_mask = domain_labels == domain_idx
        if np.any(domain_mask):
            domain_weights = weights[domain_mask]
            logger.info(
                f"   Domain {domain_idx}: {np.sum(domain_mask)} samples, "
                f"weight range [{domain_weights.min():.3f}, {domain_weights.max():.3f}], "
                f"mean weight {domain_weights.mean():.3f}"
            )

            # Analyze weight distribution
            low_weight_pct = np.mean(domain_weights < 0.5) * 100
            high_weight_pct = np.mean(domain_weights > 2.0) * 100
            logger.info(
                f"     - {low_weight_pct:.1f}% samples have weight < 0.5 (very different)"
            )
            logger.info(
                f"     - {high_weight_pct:.1f}% samples have weight > 2.0 (very similar)"
            )

    # Optional: Filter out low-quality external samples
    if filter_low_quality and len(dataframes) > 1:
        # Identify samples to keep
        keep_mask = np.ones(len(weights), dtype=bool)

        # Always keep reference domain samples
        keep_mask[domain_labels == 0] = True

        # Filter external domain samples by weight threshold
        for domain_idx in range(1, len(dataframes)):
            domain_mask = domain_labels == domain_idx
            low_weight_mask = weights < weight_threshold
            remove_mask = domain_mask & low_weight_mask
            keep_mask[remove_mask] = False

            removed_count = np.sum(remove_mask)
            total_count = np.sum(domain_mask)
            if removed_count > 0:
                logger.info(
                    f"   🚫 Filtered {removed_count}/{total_count} ({removed_count / total_count * 100:.1f}%) "
                    f"low-quality samples from domain {domain_idx}"
                )

        # Apply filtering
        combined_df = combined_df[keep_mask].reset_index(drop=True)
        weights = weights[keep_mask]

        logger.info(
            "   ✅ Kept {len(combined_df)} high-quality samples after filtering"
        )

    logger.info("Created domain-balanced dataset with {len(combined_df)} samples")
    logger.info(
        f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}"
    )

    return combined_df, weights


def save_best_trial_params(study, model_name, params_dir="best_params"):
    """Save the best trial parameters to a JSON file."""
    os.makedirs(params_dir, exist_ok=True)
    best_params = study.best_trial.params
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    with open(filepath, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Saved best parameters for {model_name} to {filepath}")
    return best_params


def load_best_trial_params(model_name, params_dir="best_params"):
    """Load the best trial parameters from a JSON file."""
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            params = json.load(f)
        logger.info("Loaded best parameters for {model_name} from {filepath}")
        return params
    else:
        logger.info("No saved parameters found for {model_name} at {filepath}")
        return None


def build_stack(trial, seed: int, wide_hp: bool) -> Pipeline:
    """Build main stacking model with CPU-only configuration"""

    # Enhanced search ranges for better performance
    if wide_hp:
        n_lo, n_hi = 600, 1200
    else:
        n_lo, n_hi = 500, 1000

    # Since we're using one-hot encoding now, no need to specify categorical columns
    # The TOP-4 solution approach uses one-hot encoding for all categorical features

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
        "n_jobs": 4,  # Use all CPU cores
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
        # categorical_feature=cat_columns,
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
        n_jobs=4,
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
        # No categorical features since we're using one-hot encoding
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
        "thread_count": -1,
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
            n_jobs=4,
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
            n_jobs=4,
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
            n_jobs=4,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=4,
    )

    return Pipeline([("stk", stk)])


def build_stack_c(trial, seed: int) -> Pipeline:
    """Build Stack C with XGBoost + CatBoost combination for diversity."""
    # Since we're using one-hot encoding now, no need to specify categorical columns

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
        "n_jobs": 4,
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
        # No categorical features since we're using one-hot encoding
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
        "thread_count": -1,
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
            n_jobs=4,
        )
    elif meta_type == "ridge":
        meta = LogisticRegression(
            C=1.0
            / trial.suggest_float("c_meta_alpha", 0.1, 10.0, log=True),  # C = 1/alpha
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=4,
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
            n_jobs=4,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=4,
    )

    return Pipeline([("stk", stk)])


def build_sklearn_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build Stack D with sklearn models with improved preprocessing."""
    # Since we're using one-hot encoding in preprocessing, we treat all features as numerical
    # Get all columns as numerical (they're all numerical after one-hot encoding)
    num_base = list(X_full.columns)

    # Use RobustScaler for all features since they're all numerical after one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_base),
            # No categorical transformer needed after one-hot encoding
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
        n_jobs=4,
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
        n_jobs=4,
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
            n_jobs=4,
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
            n_jobs=4,
        )
    elif meta_type == "lgb":
        meta = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("meta_lgb_n", 100, 300),
            learning_rate=trial.suggest_float("meta_lgb_lr", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("meta_lgb_depth", 3, 8),
            random_state=seed,
            objective="binary",
            verbose=-1,
            n_jobs=4,
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
            n_jobs=4,
        )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("rf", rf_clf), ("et", et_clf), ("hgb", hgb_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=4,
    )

    return Pipeline([("stk", stk)])


def build_neural_stack(trial, seed: int, X_full: pd.DataFrame) -> Pipeline:
    """Build Neural Network Stack with diverse architectures."""
    # Since we already use one-hot encoding in preprocessing, all features are numerical
    num_base = list(X_full.columns)

    # Use RobustScaler for all features since they're all numerical after preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_base),
            # No categorical transformer needed - already one-hot encoded
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
            C=1.0
            / trial.suggest_float(
                "neural_meta_alpha", 0.1, 10.0, log=True
            ),  # C = 1/alpha
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
    """Build a stack trained on noisy labels for regularization."""
    # Same parameters as build_stack but with different seed for noise
    n_lo, n_hi = 500, 1000
    # No categorical columns since we use one-hot encoding in preprocessing

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
        "n_jobs": -1,
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
        # categorical_feature=cat_columns,
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
        n_jobs=4,
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
        # No categorical features since we use one-hot encoding
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
        "thread_count": -1,
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
        n_jobs=4,
    )

    skf_inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=skf_inner,
        n_jobs=4,
    )

    return Pipeline([("stk", stk)])


def make_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, wide_hp: bool, sample_weights=None
):
    """Create an objective function for Optuna to optimize stacking models."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_stack(trial, seed=seed, wide_hp=wide_hp)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_stack_c_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for Stack C (XGBoost + CatBoost)."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_stack_c(trial, seed=seed)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_sklearn_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for sklearn-based Stack D."""

    def _obj(trial):
        try:
            # Build the stacking model
            model = build_sklearn_stack(trial, seed=seed, X_full=X)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_neural_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, sample_weights=None
):
    """Create an objective function for neural network Stack E."""

    def _obj(trial):
        try:
            # Build the neural network stacking model
            model = build_neural_stack(trial, seed=seed, X_full=X)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def make_noisy_stack_objective(
    X: pd.DataFrame, y: pd.Series, seed: int, noise_rate: float, sample_weights=None
):
    """Create an objective function for noisy label Stack F."""

    def _obj(trial):
        try:
            # Build the noisy stacking model
            model = build_noisy_stack(trial, seed=seed, noise_rate=noise_rate)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # Add label noise to training data
                y_train_noisy = add_label_noise(
                    y_train, noise_rate=noise_rate, random_state=seed + fold
                )

                # Fit model
                model.fit(X_train, y_train_noisy)

                # Predict and score (on clean validation labels)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

                # Report intermediate score for pruning
                trial.report(score, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        except Exception as e:
            # If anything goes wrong, prune the trial
            raise optuna.TrialPruned()

    return _obj


def oof_probs(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    sample_weights=None,
):
    """Generate out-of-fold predictions for ensemble blending."""
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info("   Fold {fold + 1}/{N_SPLITS}")

        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Build and fit model
        model = model_builder()
        model.fit(X_train, y_train)

        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds


def oof_probs_noisy(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    noise_rate: float,
    sample_weights=None,
):
    """Generate out-of-fold predictions for noisy label ensemble."""
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info("   Fold {fold + 1}/{N_SPLITS} (with {noise_rate:.1%} label noise)")

        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Add label noise to training data
        y_train_noisy = add_label_noise(
            y_train, noise_rate=noise_rate, random_state=RND + fold
        )

        # Build and fit model
        model = model_builder()
        model.fit(X_train, y_train_noisy)

        # Out-of-fold predictions (on clean validation data)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds


def improved_blend_obj(trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_true):
    """Improved blending objective with constraints and regularization."""
    # Sample blend weights
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0)
    w3 = trial.suggest_float("w3", 0.0, 1.0)
    w4 = trial.suggest_float("w4", 0.0, 1.0)
    w5 = trial.suggest_float("w5", 0.0, 1.0)
    w6 = trial.suggest_float("w6", 0.0, 1.0)

    # Normalize weights
    weights = np.array([w1, w2, w3, w4, w5, w6])
    weights = weights / np.sum(weights)

    # Calculate blended predictions
    blended = (
        weights[0] * oof_A
        + weights[1] * oof_B
        + weights[2] * oof_C
        + weights[3] * oof_D
        + weights[4] * oof_E
        + weights[5] * oof_F
    )

    # Convert to binary predictions
    y_pred = (blended >= 0.5).astype(int)

    # Calculate accuracy
    score = accuracy_score(y_true, y_pred)

    # Store normalized weights in trial attributes
    trial.set_user_attr("weights", weights.tolist())

    return score


def main():
    """Main execution function"""
    logger.info("🎯 Six-Stack Personality Classification Pipeline (CPU-Only)")
    logger.info("=" * 60)

    # Load data using TOP-4 solution merge strategy
    df_tr, df_te, submission = load_data_with_external_merge()

    # Preprocess data with TOP-4 solution approach (do this first)
    X_full, X_test, y_full, le = prep(df_tr, df_te)

    # Apply new data augmentation after preprocessing
    X_full, y_full = apply_data_augmentation(X_full, y_full)

    # Set up pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    # Initial parameters for warm start - Model A (Best: 0.9698770414156341)
    prep_parameters_a = {
        "xgb_grow": "depthwise",
        "xgb_n": 945,
        "xgb_lr": 0.015202638535985058,
        "xgb_d": 9,
        "xgb_sub": 0.852046198262209,
        "xgb_col": 0.9480247550215684,
        "xgb_alpha": 0.000379569597468035,
        "xgb_lambda": 8.425575953853357,
        "xgb_gamma": 2.9680749905938644,
        "xgb_min_child": 11,
        "lgb_n": 802,
        "lgb_lr": 0.16508685948157298,
        "lgb_d": 2,
        "lgb_sub": 0.7036333592082553,
        "lgb_col": 0.7302718535301005,
        "lgb_leaves": 159,
        "lgb_min_child": 53,
        "lgb_min_weight": 0.0009564176415918027,
        "lgb_alpha": 19.44913137025784,
        "lgb_lambda": 0.0023439694833987984,
        "lgb_cat_smooth": 36,
        "lgb_cat_l2": 14.295504457172747,
        "lgb_min_data_bin": 2,
        "lgb_path_smooth": 0.1010614204894173,
        "cat_bootstrap": "MVS",
        "cat_n": 646,
        "cat_lr": 0.031212713245561453,
        "cat_d": 10,
        "cat_l2": 1.577368711730953,
        "cat_rs": 6.85311322571503,
        "cat_leaf_iters": 7,
        "cat_grow": "Lossguide",
        "cat_min_data": 7,
        "cat_border_count": 227,
        "cat_subsample": 0.7510380706434205,
        "meta_type": "xgb",
    }

    # Model B (Best: 0.9700928201047561)
    prep_parameters_b = {
        "xgb_grow": "lossguide",
        "xgb_n": 1131,
        "xgb_lr": 0.05157765001330079,
        "xgb_d": 11,
        "xgb_sub": 0.8872877080452362,
        "xgb_col": 0.7795741489732473,
        "xgb_alpha": 0.00902833328479989,
        "xgb_lambda": 8.61136334496091,
        "xgb_gamma": 7.804614543983024,
        "xgb_min_child": 13,
        "xgb_leaves": 126,
        "lgb_n": 854,
        "lgb_lr": 0.17770038584473272,
        "lgb_d": 10,
        "lgb_sub": 0.5568110142494049,
        "lgb_col": 0.5261568302477544,
        "lgb_leaves": 188,
        "lgb_min_child": 46,
        "lgb_min_weight": 0.00038335518811955654,
        "lgb_alpha": 0.07236992480642132,
        "lgb_lambda": 0.07212731897883105,
        "lgb_cat_smooth": 67,
        "lgb_cat_l2": 8.305236603222985,
        "lgb_min_data_bin": 11,
        "lgb_path_smooth": 0.10035424976991908,
        "cat_bootstrap": "MVS",
        "cat_n": 880,
        "cat_lr": 0.015993755307381854,
        "cat_d": 11,
        "cat_l2": 12.450962287776338,
        "cat_rs": 16.72836530669729,
        "cat_leaf_iters": 11,
        "cat_grow": "Lossguide",
        "cat_min_data": 18,
        "cat_border_count": 283,
        "cat_subsample": 0.8553510476562626,
        "meta_type": "logistic",
        "meta_log_c": 6.459864162187188,
    }

    # Model C (Best: 0.9700386786870816)
    prep_parameters_c = {
        "xgb_grow": "lossguide",
        "xgb_n": 448,
        "xgb_lr": 0.15750641449223535,
        "xgb_d": 11,
        "xgb_sub": 0.6219348595208609,
        "xgb_col": 0.6237860243259894,
        "xgb_alpha": 0.046983945094351026,
        "xgb_lambda": 9.70175049299882,
        "xgb_gamma": 3.931277261316059,
        "xgb_min_child": 6,
        "xgb_leaves": 66,
        "cat_bootstrap": "Bayesian",
        "cat_n": 770,
        "cat_lr": 0.0360746361642475,
        "cat_d": 7,
        "cat_l2": 6.281663165541404,
        "cat_rs": 6.061917523212558,
        "cat_leaf_iters": 14,
        "cat_grow": "SymmetricTree",
        "cat_min_data": 18,
        "cat_border_count": 207,
        "cat_temp": 0.6510949621686273,
        "c_meta_type": "logistic",
        "c_meta_c": 8.305622416200901,
    }

    # Model D (Best: 0.9698232060463503)
    prep_parameters_d = {
        "rf_n": 508,
        "rf_depth": 21,
        "rf_min_split": 5,
        "rf_min_leaf": 1,
        "rf_max_features": None,
        "rf_class_weight": None,
        "et_n": 950,
        "et_depth": 39,
        "et_min_split": 6,
        "et_min_leaf": 5,
        "et_max_features": "log2",
        "et_class_weight": None,
        "hgb_n": 800,
        "hgb_lr": 0.08007953809056918,
        "hgb_depth": 8,
        "hgb_min_leaf": 16,
        "hgb_l2": 0.6740832112808466,
        "meta_type": "xgb",
        "meta_xgb_n": 246,
        "meta_xgb_lr": 0.06589834679918967,
        "meta_xgb_depth": 6,
    }

    # Model E (Best: 0.9698230165878229)
    prep_parameters_e = {
        "mlp1_h1": 106,
        "mlp1_h2": 86,
        "mlp1_h3": 50,
        "mlp1_lr": 0.0006306913268514695,
        "mlp1_alpha": 0.02019413520280658,
        "mlp1_iter": 1579,
        "mlp2_h1": 342,
        "mlp2_lr": 0.0010947538981036953,
        "mlp2_alpha": 0.00016513079871827957,
        "mlp2_iter": 621,
        "svm_c": 1.17311282154475,
        "svm_gamma": "auto",
        "svm_kernel": "rbf",
        "nb_var_smooth": 9.553142993144058e-09,
        "neural_meta_type": "logistic",
        "neural_meta_c": 8.063615330559422,
    }

    # Model F (Best: 0.9700387661294789)
    prep_parameters_f = {
        "xgb_grow": "depthwise",
        "xgb_n": 651,
        "xgb_lr": 0.19980229596023016,
        "xgb_d": 7,
        "xgb_sub": 0.9646027011006114,
        "xgb_col": 0.7110655132254216,
        "xgb_alpha": 0.005890889327182349,
        "xgb_lambda": 14.858206311292482,
        "xgb_gamma": 0.49066560158731787,
        "xgb_min_child": 3,
        "lgb_n": 929,
        "lgb_lr": 0.013667186190795077,
        "lgb_d": -1,
        "lgb_sub": 0.9125801588716573,
        "lgb_col": 0.9311697942047846,
        "lgb_leaves": 110,
        "lgb_min_child": 72,
        "lgb_min_weight": 0.01456398702838513,
        "lgb_alpha": 1.051308135053416,
        "lgb_lambda": 0.04114612588000621,
        "lgb_cat_smooth": 198,
        "lgb_cat_l2": 14.631872253652649,
        "lgb_min_data_bin": 43,
        "lgb_path_smooth": 0.15361398097734175,
        "cat_bootstrap": "Bayesian",
        "cat_n": 679,
        "cat_lr": 0.03117224079638828,
        "cat_d": 10,
        "cat_l2": 8.666707732672174,
        "cat_rs": 1.7502250623280302,
        "cat_leaf_iters": 10,
        "cat_grow": "Lossguide",
        "cat_min_data": 18,
        "cat_border_count": 176,
        "cat_temp": 0.8741016355692319,
        "meta_log_c": 0.03105340492419464,
    }

    # Train 6 stacks
    logger.info("\n🔍 Training 6 specialized stacks...")

    # Stack E - Neural networks
    logger.info("Training Stack E - Neural Networks...")
    study_E = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    # Try to load and enqueue best parameters from previous run
    best_params_E = load_best_trial_params("stack_E")
    if best_params_E:
        study_E.enqueue_trial(best_params_E)
    study_E.optimize(
        make_neural_stack_objective(X_full, y_full, seed=7777, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    # Save best parameters
    save_best_trial_params(study_E, "stack_E")

    # Stack F - Noisy labels
    logger.info("Training Stack F - Noisy Labels...")
    study_F = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    # Try to load and enqueue best parameters from previous run
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
    # Save best parameters
    save_best_trial_params(study_F, "stack_F")

    # Stack A - Traditional ML (narrow hyperparameters)
    logger.info("Training Stack A...")
    study_A = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    # Try to load and enqueue best parameters from previous run
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
    # Save best parameters
    save_best_trial_params(study_A, "stack_A")

    # Stack B - Traditional ML (wide hyperparameters)
    logger.info("Training Stack B...")
    study_B = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )
    # Try to load and enqueue best parameters from previous run
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
    # Save best parameters
    save_best_trial_params(study_B, "stack_B")

    # Stack C - XGBoost + CatBoost
    logger.info("Training Stack C...")
    study_C = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    # Try to load and enqueue best parameters from previous run
    best_params_C = load_best_trial_params("stack_C")
    if best_params_C:
        study_C.enqueue_trial(best_params_C)
    study_C.optimize(
        make_stack_c_objective(X_full, y_full, seed=1337, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    # Save best parameters
    save_best_trial_params(study_C, "stack_C")

    # Stack D - Sklearn models
    logger.info("Training Stack D - Sklearn...")
    study_D = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )
    # Try to load and enqueue best parameters from previous run
    best_params_D = load_best_trial_params("stack_D")
    if best_params_D:
        study_D.enqueue_trial(best_params_D)
    study_D.optimize(
        make_sklearn_stack_objective(X_full, y_full, seed=9999, sample_weights=None),
        n_trials=N_TRIALS_STACK,
        show_progress_bar=True,
    )
    # Save best parameters
    save_best_trial_params(study_D, "stack_D")

    # Create model builders
    def builder_A():
        return build_stack(study_A.best_trial, seed=RND, wide_hp=False)

    def builder_B():
        return build_stack(study_B.best_trial, seed=2024, wide_hp=True)

    def builder_C():
        return build_stack_c(study_C.best_trial, seed=1337)

    def builder_D():
        return build_sklearn_stack(study_D.best_trial, seed=9999, X_full=X_full)

    def builder_E():
        return build_neural_stack(study_E.best_trial, seed=7777, X_full=X_full)

    def builder_F():
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
    study_improved = optuna.create_study(
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
    study_improved.enqueue_trial(best_blend_params)

    def blend_objective(trial):
        return improved_blend_obj(
            trial, oof_A, oof_B, oof_C, oof_D, oof_E, oof_F, y_full
        )

    study_improved.optimize(
        blend_objective, n_trials=N_TRIALS_BLEND, show_progress_bar=True
    )

    # Extract best weights
    best_trial = study_improved.best_trial
    best_weights = np.array(best_trial.user_attrs["weights"])
    wA, wB, wC, wD, wE, wF = best_weights

    logger.info(
        f"\n🏆 Best blend weights: wA={wA:.3f}, wB={wB:.3f}, wC={wC:.3f}, wD={wD:.3f}, wE={wE:.3f}, wF={wF:.3f}"
    )
    logger.info(f"Best CV score: {study_improved.best_value:.6f}")

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

    # 🔮 PSEUDO-LABELING: Generate test predictions for pseudo-labeling
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
    submission_df = pd.DataFrame({"id": submission.id, "Personality": personality})

    # Save results
    output_file = "six_stack_personality_predictions_with_external.csv"
    submission_df.to_csv(output_file, index=False)

    logger.info("\n✅ Predictions saved to '{output_file}'")
    logger.info("📊 Final submission shape: {submission_df.shape}")
    logger.info("🎉 Six-stack ensemble pipeline completed successfully!")

    # Print summary
    logger.info("\n📋 Summary:")
    logger.info("   - Combined training data: {len(df_tr):,} samples")
    logger.info("   - External data merged as features using TOP-4 solution approach")
    logger.info("   - 6 specialized stacks trained")
    logger.info("   - Best ensemble CV score: {study_improved.best_value:.6f}")
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


if __name__ == "__main__":
    main()
