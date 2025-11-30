"""
Data preprocessing functions for the personality classification pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .utils import get_logger

logger = get_logger(__name__)


def prep(
    df_tr: pd.DataFrame, df_te: pd.DataFrame, tgt="Personality", idx="id"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the training and test datasets with advanced competitive approach.

    Args:
        df_tr: Training dataframe
        df_te: Test dataframe
        tgt: Target column name
        idx: Index column name

    Returns:
        Tuple of (X_train, X_test, y_train, label_encoder)
    """
    logger.info("Preprocessing data with advanced competitive approach...")

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

    # Use advanced correlation-based imputation
    logger.info("Performing TOP-4 correlation-based imputation...")

    # Extract and encode target variable BEFORE combining data
    le_tgt = LabelEncoder()
    ytr = pd.Series(le_tgt.fit_transform(df_tr[tgt]), name=tgt)

    # Remove target column from training data before combining
    df_tr_no_target = df_tr.drop(columns=[tgt])

    # Combine train and test for imputation (as in advanced approach)
    ntrain = len(df_tr_no_target)
    all_data = pd.concat([df_tr_no_target, df_te], ignore_index=True)

    def fill_missing_by_quantile_group(
        df, group_source_col, target_col, quantiles=None
    ):
        """Fill missing values using correlation-based grouping (from advanced approach)"""
        if quantiles is None:
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
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

    # Sequential imputation based on correlations (from advanced approach)
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

    # Handle categorical features with "Unknown" (from advanced approach)
    categorical_cols = ["Stage_fear", "Drained_after_socializing", "match_p"]
    for col in categorical_cols:
        if col in all_data.columns:
            # Fill missing values with 'Unknown'
            all_data[col] = all_data[col].fillna("Unknown")

    # Apply one-hot encoding for categorical features using OneHotEncoder (advanced approach)
    logger.info("Applying one-hot encoding for categorical features...")

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
            f"   Encoded {len(existing_categorical_cols)} categorical features into {len(feature_names)} binary features"
        )
    else:
        logger.warning("   No categorical features found to encode")

    # Fill any remaining missing values
    logger.info("Filling any remaining missing values...")

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

    logger.info("Preprocessing completed with advanced competitive approach")
    return df_tr, df_te, ytr, le_tgt


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
        f"🔮 Adding pseudo-labels with confidence threshold {confidence_threshold}"
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

    logger.info(f"   Found {n_high_conf} high-confidence predictions")
    logger.info(f"   Maximum allowed pseudo-samples: {max_pseudo_samples}")

    if n_high_conf > 0:
        # Limit pseudo-samples to avoid overfitting
        if n_high_conf > max_pseudo_samples:
            logger.info(f"   Limiting to {max_pseudo_samples} most confident samples")
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

        logger.info(f"   Added {len(y_pseudo)} pseudo-labels to training data")
        logger.info(
            f"   Pseudo-label distribution: Class 0: {pseudo_stats['pseudo_class_0']}, Class 1: {pseudo_stats['pseudo_class_1']}"
        )
        logger.info(f"   Mean confidence: {pseudo_stats['mean_confidence']:.4f}")
        logger.info(f"   Training data: {len(X_full)} → {len(X_combined)} samples")

        return X_combined, y_combined, pseudo_stats
    else:
        logger.info("   No high-confidence predictions found")
        pseudo_stats = {
            "n_pseudo_added": 0,
            "original_size": len(X_full),
            "final_size": len(X_full),
        }
        return X_full, y_full, pseudo_stats


def create_domain_balanced_dataset(
    dataframes: list[pd.DataFrame],
    target_column: str = "Personality",
    domain_names: list[str] | None = None,
    random_state: int = 42,
    filter_low_quality: bool = True,
    weight_threshold: float = 0.2,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    logger.info("Computing domain weights for distribution alignment...")

    # Combine dataframes with domain labels
    combined_data = []
    domain_labels = []

    for i, df in enumerate(dataframes):
        combined_data.append(df.copy())
        domain_labels.extend([i] * len(df))

    # Create combined dataset
    combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
    domain_labels = np.array(domain_labels).tolist()  # Convert to list for consistency

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
    logger.info("Domain weighting summary:")
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
                    f"   Filtered {removed_count}/{total_count} ({removed_count / total_count * 100:.1f}%) "
                    f"low-quality samples from domain {domain_idx}"
                )

        # Apply filtering
        combined_df = combined_df[keep_mask].reset_index(drop=True)
        weights = weights[keep_mask]

        logger.info(f"   Kept {len(combined_df)} high-quality samples after filtering")

    logger.info(f"Created domain-balanced dataset with {len(combined_df)} samples")
    logger.info(
        f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}"
    )

    return combined_df, weights
