"""
Data augmentation functions for the personality classification pipeline.
"""

import platform
import signal
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import (
    AUGMENTATION_METHOD,
    AUGMENTATION_RATIO,
    ENABLE_DATA_AUGMENTATION,
    IMBLEARN_AVAILABLE,
    RND,
    SDV_AVAILABLE,
)
from .utils import get_logger


# Conditional imports
if SDV_AVAILABLE:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

if IMBLEARN_AVAILABLE:
    from imblearn.over_sampling import SMOTENC

logger = get_logger(__name__)


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

        # Fit the synthesizer with timeout protection
        logger.info("   🏋️ Training synthesizer...")

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
            f"⚠️ SDV augmentation failed: {e!s}, falling back to simple augmentation"
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
            f"⚠️ SMOTENC augmentation failed: {e!s}, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, 0.1)


def apply_data_augmentation(X_train, y_train):
    """Apply the configured data augmentation method"""
    if not ENABLE_DATA_AUGMENTATION:
        logger.info("�� Data augmentation disabled")
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
