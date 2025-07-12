"""
Enhanced data augmentation functions for the personality classification pipeline.
Features: Adaptive method selection, quality control, diversity checking, class balancing.
"""

import platform
import signal
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .config import (
    IMBLEARN_AVAILABLE,
    RND,
    SDV_AVAILABLE,
    AugmentationConfig,
    AugmentationMethod,
)
from .utils import get_logger


# Conditional imports
if SDV_AVAILABLE:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

if IMBLEARN_AVAILABLE:
    from imblearn.over_sampling import SMOTENC

logger = get_logger(__name__)


def analyze_data_characteristics(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Analyze data to determine optimal augmentation strategy."""
    n_samples, n_features = X_train.shape

    # Class balance analysis
    class_counts = y_train.value_counts()
    class_balance_ratio = class_counts.min() / class_counts.max()

    # Feature type analysis
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    categorical_ratio = len(categorical_cols) / n_features

    # Data complexity analysis
    rf_temp = RandomForestClassifier(n_estimators=10, random_state=RND, n_jobs=1)
    rf_temp.fit(X_train, y_train)
    feature_importance_std = np.std(rf_temp.feature_importances_)

    characteristics = {
        "n_samples": n_samples,
        "n_features": n_features,
        "class_balance_ratio": class_balance_ratio,
        "categorical_ratio": categorical_ratio,
        "feature_complexity": feature_importance_std,
        "is_small_dataset": n_samples < 5000,
        "is_imbalanced": class_balance_ratio < 0.5,
        "is_highly_categorical": categorical_ratio > 0.3,
    }

    logger.info(f"📊 Data characteristics: {characteristics}")
    return characteristics


def adaptive_augmentation_selection(
    characteristics: dict,
) -> tuple[AugmentationMethod, float]:
    """Select optimal augmentation method based on data characteristics.

    Now respects configuration ratios from AugmentationConfig instead of hardcoded values.
    """
    # Use configuration ratios as base, with adaptive scaling
    base_ratio = AugmentationConfig.BASE_AUGMENTATION_RATIO.value
    min_ratio = AugmentationConfig.MIN_AUGMENTATION_RATIO.value
    max_ratio = AugmentationConfig.MAX_AUGMENTATION_RATIO.value

    logger.info(
        f"   📊 Adaptive selection using config ratios: base={base_ratio}, min={min_ratio}, max={max_ratio}"
    )

    if characteristics["is_small_dataset"] and characteristics["is_imbalanced"]:
        # Small imbalanced dataset - use higher ratio for SMOTENC
        adaptive_ratio = min(base_ratio * 1.5, max_ratio)  # 150% of base, capped at max
        logger.info(
            f"   🎯 Small imbalanced dataset detected → SMOTENC with ratio {adaptive_ratio:.3f} (base*1.5)"
        )
        return AugmentationMethod.SMOTENC, adaptive_ratio

    elif characteristics["is_highly_categorical"]:
        # High categorical ratio - use moderate ratio for SDV Copula
        adaptive_ratio = min(base_ratio * 0.8, max_ratio)  # 80% of base
        logger.info(
            f"   🎯 High categorical features detected → SDV_COPULA with ratio {adaptive_ratio:.3f} (base*0.8)"
        )
        return AugmentationMethod.SDV_COPULA, adaptive_ratio

    elif characteristics["n_samples"] > 10000 and not characteristics["is_imbalanced"]:
        # Large balanced dataset - use conservative ratio for ensemble
        adaptive_ratio = max(base_ratio * 0.5, min_ratio)  # 50% of base, at least min
        logger.info(
            f"   🎯 Large balanced dataset detected → MIXED_ENSEMBLE with ratio {adaptive_ratio:.3f} (base*0.5)"
        )
        return AugmentationMethod.MIXED_ENSEMBLE, adaptive_ratio

    elif characteristics["class_balance_ratio"] < 0.3:
        # Severe imbalance - use maximum ratio for class balancing
        adaptive_ratio = max_ratio  # Use maximum configured ratio
        logger.info(
            f"   🎯 Severe class imbalance detected → CLASS_BALANCED with ratio {adaptive_ratio:.3f} (max ratio)"
        )
        return AugmentationMethod.CLASS_BALANCED, adaptive_ratio

    else:
        # Default to SDV Copula with base ratio
        adaptive_ratio = base_ratio
        logger.info(
            f"   🎯 Default case → SDV_COPULA with ratio {adaptive_ratio:.3f} (base ratio)"
        )
        return AugmentationMethod.SDV_COPULA, adaptive_ratio


def tvae_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series, augment_ratio: float = 0.05
) -> tuple[pd.DataFrame, pd.Series]:
    """TVAE-based augmentation with better stability than CTGAN."""
    if not SDV_AVAILABLE:
        logger.warning("⚠️ SDV not available, falling back to simple augmentation")
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)

    try:
        logger.info("   🧠 Using TVAE augmentation...")

        # Prepare data
        combined_data = X_train.copy()
        combined_data["Personality"] = y_train

        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(combined_data)

        # Update metadata for categorical columns
        for col in combined_data.columns:
            if col.startswith(
                ("Stage_fear_", "Drained_after_socializing_", "match_p_")
            ):
                metadata.update_column(col, sdtype="categorical")
        metadata.update_column("Personality", sdtype="categorical")

        # Configure TVAE with conservative settings for stability
        synthesizer = TVAESynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=100,
            verbose=False,
            cuda=False,  # Force CPU for stability
        )

        synthesizer.fit(combined_data)

        # Generate samples
        n_samples = max(10, int(len(X_train) * augment_ratio))
        synthetic_data = synthesizer.sample(num_rows=n_samples)

        augmented_X = synthetic_data.drop("Personality", axis=1)
        augmented_y = synthetic_data["Personality"]

        return augmented_X, augmented_y

    except Exception as e:
        logger.warning(f"⚠️ TVAE failed: {e}, falling back to Copula")
        return sdv_augmentation(X_train, y_train, "copula", augment_ratio)


def class_balanced_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series, target_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.Series]:
    """Intelligent class balancing with multiple methods."""
    logger.info("   ⚖️ Using class-balanced augmentation...")

    class_counts = y_train.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Calculate how many samples needed for target balance
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    target_minority_count = int(majority_count * target_ratio)
    samples_needed = max(0, target_minority_count - minority_count)

    if samples_needed == 0:
        logger.info("   ✅ Classes already balanced, no augmentation needed")
        return pd.DataFrame(), pd.Series(dtype=y_train.dtype)

    # Filter minority class data
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask]
    y_minority = y_train[minority_mask]

    # Use ensemble of methods for better diversity
    methods = [("smotenc", 0.6), ("copula", 0.4)]

    all_augmented_X = []
    all_augmented_y = []

    for method, ratio in methods:
        method_samples = int(samples_needed * ratio)
        method_ratio = method_samples / len(X_minority) if len(X_minority) > 0 else 0

        try:
            if method == "smotenc":
                aug_X, aug_y = smotenc_augmentation(
                    X_minority, y_minority, method_ratio
                )
            else:  # copula
                aug_X, aug_y = sdv_augmentation(
                    X_minority, y_minority, "copula", method_ratio
                )

            if len(aug_X) > 0:
                all_augmented_X.append(aug_X)
                all_augmented_y.append(aug_y)

        except Exception as e:
            logger.warning(f"⚠️ Method {method} failed: {e}")

    if all_augmented_X:
        combined_X = pd.concat(all_augmented_X, ignore_index=True)
        combined_y = pd.concat(all_augmented_y, ignore_index=True)
        return combined_X, combined_y

    return pd.DataFrame(), pd.Series(dtype=y_train.dtype)


def mixed_ensemble_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series, augment_ratio: float = 0.05
) -> tuple[pd.DataFrame, pd.Series]:
    """Ensemble multiple methods for maximum diversity."""
    logger.info("   🎭 Using mixed ensemble augmentation...")

    methods = [("copula", 0.5), ("smotenc", 0.5)]

    all_augmented_X = []
    all_augmented_y = []

    for method, ratio in methods:
        method_ratio = augment_ratio * ratio

        try:
            if method == "copula":
                aug_X, aug_y = sdv_augmentation(
                    X_train, y_train, "copula", method_ratio
                )
            elif method == "smotenc":
                aug_X, aug_y = smotenc_augmentation(X_train, y_train, method_ratio)

            if len(aug_X) > 0:
                all_augmented_X.append(aug_X)
                all_augmented_y.append(aug_y)

        except Exception as e:
            logger.warning(f"⚠️ Method {method} failed: {e}")

    if all_augmented_X:
        combined_X = pd.concat(all_augmented_X, ignore_index=True)
        combined_y = pd.concat(all_augmented_y, ignore_index=True)
        return combined_X, combined_y

    return pd.DataFrame(), pd.Series(dtype=y_train.dtype)


def enhanced_quality_filtering(
    original_X: pd.DataFrame,
    original_y: pd.Series,
    synthetic_X: pd.DataFrame,
    synthetic_y: pd.Series,
    threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.Series]:
    """Multi-metric quality filtering for synthetic data."""
    logger.info("   🔍 Enhanced quality filtering...")

    if len(synthetic_X) == 0:
        return synthetic_X, synthetic_y

    # 1. Classifier confidence
    rf = RandomForestClassifier(n_estimators=20, random_state=RND, n_jobs=1)
    rf.fit(original_X, original_y)

    synthetic_probas = rf.predict_proba(synthetic_X)
    confidence_scores = np.max(synthetic_probas, axis=1)

    # 2. Feature distribution similarity
    distribution_scores = []

    for col in original_X.columns:
        if original_X[col].dtype in ["int64", "float64"]:
            try:
                ks_stat, _ = ks_2samp(
                    original_X[col].dropna(), synthetic_X[col].dropna()
                )
                similarity = 1 - ks_stat  # Convert to similarity score
                distribution_scores.append(similarity)
            except Exception:
                distribution_scores.append(0.5)  # Default neutral score

    avg_distribution_similarity = (
        np.mean(distribution_scores) if distribution_scores else 1.0
    )

    # 3. Combined quality score
    quality_scores = (confidence_scores * 0.7) + (avg_distribution_similarity * 0.3)

    # Filter based on combined quality
    high_quality_mask = quality_scores >= threshold

    filtered_X = synthetic_X[high_quality_mask]
    filtered_y = synthetic_y[high_quality_mask]

    logger.info(
        f"   ✨ Quality filtering: {len(filtered_X)}/{len(synthetic_X)} samples kept"
    )
    logger.info(f"   📊 Avg quality score: {np.mean(quality_scores):.3f}")

    return filtered_X, filtered_y


def diversity_check(
    original_X: pd.DataFrame, synthetic_X: pd.DataFrame, threshold: float = 0.8
) -> bool:
    """Check if synthetic data adds meaningful diversity."""
    if len(synthetic_X) == 0:
        return True

    # Sample subset for efficiency
    sample_size = min(1000, len(original_X), len(synthetic_X))
    orig_sample = original_X.sample(n=sample_size, random_state=RND)
    synth_sample = synthetic_X.sample(n=sample_size, random_state=RND)

    # Calculate feature-wise diversity
    diversity_scores = []
    for col in orig_sample.columns:
        if orig_sample[col].dtype in ["int64", "float64"]:
            orig_std = orig_sample[col].std()
            synth_std = synth_sample[col].std()
            # Diversity score based on standard deviation difference
            diversity = min(synth_std / (orig_std + 1e-6), 2.0)  # Cap at 2.0
            diversity_scores.append(diversity)

    avg_diversity = np.mean(diversity_scores) if diversity_scores else 1.0
    is_diverse = avg_diversity >= threshold

    logger.info(
        f"   🌈 Diversity score: {avg_diversity:.3f} ({'✅ Pass' if is_diverse else '❌ Fail'})"
    )
    return is_diverse


def simple_mixed_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series, augment_ratio: float = 0.05
) -> tuple[pd.DataFrame, pd.Series]:
    """Simple data augmentation for mixed numerical/categorical features."""

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
        return pd.DataFrame(), pd.Series(dtype=y_train.dtype)

    n_augment = int(len(minority_X) * augment_ratio)
    if n_augment == 0:
        return pd.DataFrame(), pd.Series(dtype=y_train.dtype)

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


def sdv_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "copula",
    augment_ratio: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series]:
    """High-quality synthetic data generation using SDV with improved CTGAN handling."""
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

        # Update metadata for categorical columns
        for col in train_data.columns:
            if col.startswith(
                ("Stage_fear_", "Drained_after_socializing_", "match_p_")
            ):
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

                # Conservative CTGAN parameters
                data_size = len(train_data)
                if data_size > 10000:
                    epochs = 5
                    batch_size = 500
                elif data_size > 5000:
                    epochs = 10
                    batch_size = 250
                else:
                    epochs = 20
                    batch_size = min(100, data_size // 8)

                try:
                    synthesizer = CTGANSynthesizer(
                        metadata,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=False,
                        cuda=False,
                        discriminator_decay=1e-5,
                        generator_decay=1e-5,
                        discriminator_steps=1,
                        pac=1,
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ CTGAN initialization failed: {e}")
                    synthesizer = GaussianCopulaSynthesizer(metadata)
        else:
            synthesizer = GaussianCopulaSynthesizer(metadata)

        # Fit with timeout protection
        logger.info("   🏋️ Training synthesizer...")

        def timeout_handler(_signum, _frame):
            raise TimeoutError("Synthesizer training timed out")

        start_time = time.time()

        if method == "ctgan":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)  # 3 minutes timeout

        try:
            synthesizer.fit(train_data)
            training_time = time.time() - start_time
            logger.info(f"   ✅ Training completed in {training_time:.1f}s")
        finally:
            if method == "ctgan":
                signal.alarm(0)

        # Generate synthetic data
        n_synthetic = int(len(train_data) * augment_ratio)
        if n_synthetic == 0:
            return pd.DataFrame(), pd.Series(dtype=y_train.dtype)

        logger.info(f"   🎲 Generating {n_synthetic} synthetic samples...")
        synthetic_data = synthesizer.sample(num_rows=n_synthetic)

        # Split back to features and target
        synthetic_X = synthetic_data.drop("Personality", axis=1)
        synthetic_y = synthetic_data["Personality"]

        logger.info(f"   ✅ Generated {len(synthetic_X)} synthetic samples")
        return synthetic_X, synthetic_y

    except TimeoutError:
        logger.warning("⚠️ Training timed out, falling back to simple augmentation")
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)
    except Exception as e:
        logger.warning(
            f"⚠️ SDV augmentation failed: {e}, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)


def smotenc_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series, augment_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE for mixed numerical/categorical data."""
    if not IMBLEARN_AVAILABLE:
        logger.warning(
            "⚠️ imbalanced-learn not available, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)

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
            f"⚠️ SMOTENC augmentation failed: {e}, falling back to simple augmentation"
        )
        return simple_mixed_augmentation(X_train, y_train, augment_ratio)


def apply_data_augmentation(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Enhanced adaptive data augmentation with quality control."""

    if not AugmentationConfig.ENABLE_DATA_AUGMENTATION.value:
        logger.info("📊 Data augmentation disabled")
        return X_train, y_train

    logger.info("📊 Applying adaptive data augmentation...")
    original_shape = X_train.shape

    # Analyze data characteristics
    characteristics = analyze_data_characteristics(X_train, y_train)

    # Select optimal method and ratio
    if AugmentationConfig.AUGMENTATION_METHOD.value == AugmentationMethod.ADAPTIVE:
        method, augment_ratio = adaptive_augmentation_selection(characteristics)
        logger.info(
            f"   🎯 Auto-selected: {method.value} with ratio {augment_ratio:.3f}"
        )
    else:
        method = AugmentationConfig.AUGMENTATION_METHOD.value
        augment_ratio = AugmentationConfig.AUGMENTATION_RATIO.value
        logger.info(
            f"   🎯 Using configured: {method.value} with ratio {augment_ratio:.3f}"
        )

    # Apply augmentation with timeout
    def timeout_handler(_signum, _frame):
        raise TimeoutError("Augmentation timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(AugmentationConfig.MAX_AUGMENTATION_TIME_SECONDS.value)

    try:
        start_time = time.time()

        # Route to appropriate method
        if method == AugmentationMethod.TVAE:
            augmented_X, augmented_y = tvae_augmentation(
                X_train, y_train, augment_ratio
            )
        elif method == AugmentationMethod.MIXED_ENSEMBLE:
            augmented_X, augmented_y = mixed_ensemble_augmentation(
                X_train, y_train, augment_ratio
            )
        elif method == AugmentationMethod.CLASS_BALANCED:
            augmented_X, augmented_y = class_balanced_augmentation(X_train, y_train)
        elif method == AugmentationMethod.SDV_COPULA:
            augmented_X, augmented_y = sdv_augmentation(
                X_train, y_train, "copula", augment_ratio
            )
        elif method == AugmentationMethod.SMOTENC:
            augmented_X, augmented_y = smotenc_augmentation(
                X_train, y_train, augment_ratio
            )
        elif method == AugmentationMethod.ADAPTIVE:
            # Fallback if adaptive wasn't handled above
            method, augment_ratio = adaptive_augmentation_selection(characteristics)
            augmented_X, augmented_y = sdv_augmentation(
                X_train, y_train, "copula", augment_ratio
            )
        else:
            augmented_X, augmented_y = simple_mixed_augmentation(
                X_train, y_train, augment_ratio
            )

        signal.alarm(0)  # Cancel timeout
        generation_time = time.time() - start_time

    except TimeoutError:
        logger.warning(
            f"⚠️ Augmentation timeout after {AugmentationConfig.MAX_AUGMENTATION_TIME_SECONDS.value}s, using SMOTENC fallback"
        )
        augmented_X, augmented_y = smotenc_augmentation(X_train, y_train, 0.03)
        signal.alarm(0)
    except Exception as e:
        logger.warning(f"⚠️ Augmentation failed: {e}, using original data")
        signal.alarm(0)
        return X_train, y_train

    # Quality control pipeline
    if len(augmented_X) > 0:
        logger.info(f"   ⏱️ Generation time: {generation_time:.1f}s")

        # Quality filtering
        if AugmentationConfig.ENABLE_QUALITY_FILTERING.value:
            augmented_X, augmented_y = enhanced_quality_filtering(
                X_train,
                y_train,
                augmented_X,
                augmented_y,
                AugmentationConfig.QUALITY_THRESHOLD.value,
            )

        # Diversity check
        if AugmentationConfig.ENABLE_DIVERSITY_CHECK.value and not diversity_check(
            X_train, augmented_X, AugmentationConfig.DIVERSITY_THRESHOLD.value
        ):
            logger.warning("   ⚠️ Low diversity detected, reducing synthetic samples")
            # Keep only top 50% most diverse samples
            keep_ratio = 0.5
            keep_count = int(len(augmented_X) * keep_ratio)
            augmented_X = augmented_X.iloc[:keep_count]
            augmented_y = augmented_y.iloc[:keep_count]

        # Final combination
        if len(augmented_X) > 0:
            X_combined = pd.concat([X_train, augmented_X], ignore_index=True)
            y_combined = pd.concat([y_train, augmented_y], ignore_index=True)

            logger.info(
                f"   ✅ Added {len(augmented_X)} high-quality synthetic samples"
            )
            logger.info(f"   📈 Data shape: {original_shape} → {X_combined.shape}")

            # Log class balance improvement
            orig_balance = y_train.value_counts().min() / y_train.value_counts().max()
            new_balance = (
                y_combined.value_counts().min() / y_combined.value_counts().max()
            )
            logger.info(f"   ⚖️ Class balance: {orig_balance:.3f} → {new_balance:.3f}")

            return X_combined, y_combined

    logger.warning("⚠️ No synthetic samples generated, using original data")
    return X_train, y_train
