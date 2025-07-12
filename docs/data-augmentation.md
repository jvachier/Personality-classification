# Data Augmentation Guide

## Overview

The Six-Stack Personality Classification Pipeline features an advanced, adaptive data augmentation system designed to improve model generalization and performance through high-quality synthetic data generation.

## Architecture

### Adaptive Strategy Selection

The pipeline automatically selects the optimal augmentation method based on dataset characteristics:

```python
def analyze_data_characteristics(X, y):
    """Analyze dataset to determine optimal augmentation strategy."""
    return {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'class_balance_ratio': min(y.value_counts()) / max(y.value_counts()),
        'categorical_ratio': (X.dtypes == 'object').sum() / len(X.columns),
        'feature_complexity': calculate_feature_complexity(X),
        'is_small_dataset': len(X) < 1000,
        'is_imbalanced': min(y.value_counts()) / max(y.value_counts()) < 0.3,
        'is_highly_categorical': (X.dtypes == 'object').sum() / len(X.columns) > 0.5
    }
```

### Decision Matrix

| Dataset Characteristics          | Recommended Method | Rationale                         |
| -------------------------------- | ------------------ | --------------------------------- |
| Small datasets (<1K samples)     | SMOTE              | Fast, proven for small data       |
| Severe imbalance (<30% minority) | ADASYN             | Adaptive sampling for minorities  |
| High categorical (>50%)          | Basic              | Simple methods for categorical    |
| Complex numerical data           | SDV Copula         | Preserves complex distributions   |
| Large balanced datasets          | SDV Copula         | Best quality for complex patterns |

## Augmentation Methods

### 1. SDV Copula (Recommended)

**Best for**: Large datasets with complex feature distributions

#### Features

- **Gaussian Copula modeling** for complex dependency structures
- **Marginal distribution preservation** for each feature
- **Correlation structure maintenance** across features
- **Fast training mode** for development/testing

#### Implementation

```python
def sdv_copula_augmentation(X, y, n_samples):
    """Generate synthetic data using SDV Gaussian Copula."""
    # Combine features and target
    data = X.copy()
    data['target'] = y

    # Configure copula synthesizer
    synthesizer = GaussianCopula(
        enforce_rounding=True,
        enforce_min_max_values=True
    )

    # Fast training for development
    if TESTING_MODE:
        synthesizer.fit(data, epochs=5)
    else:
        synthesizer.fit(data)

    # Generate synthetic samples
    synthetic_data = synthesizer.sample(n_samples)

    # Separate features and target
    X_synthetic = synthetic_data.drop('target', axis=1)
    y_synthetic = synthetic_data['target']

    return X_synthetic, y_synthetic
```

#### Quality Controls

- **Distribution similarity** validation
- **Correlation preservation** checking
- **Statistical tests** for authenticity
- **Outlier detection** and removal

### 2. SMOTE (Synthetic Minority Oversampling)

**Best for**: Small to medium datasets with class imbalance

#### Features

- **k-NN based synthesis** for minority class
- **Interpolation strategy** between nearest neighbors
- **Configurable k parameter** (auto-optimized)
- **BorderlineSMOTE variant** for difficult cases

#### Implementation

```python
def smote_augmentation(X, y, n_samples):
    """SMOTE-based synthetic sample generation."""
    # Determine optimal k based on minority class size
    minority_size = min(y.value_counts())
    k_neighbors = min(5, minority_size - 1)

    if k_neighbors < 1:
        # Fallback to basic augmentation
        return basic_augmentation(X, y, n_samples)

    # Configure SMOTE
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=k_neighbors,
        random_state=RND
    )

    # Apply oversampling
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Extract only the new synthetic samples
    original_size = len(X)
    X_synthetic = X_resampled[original_size:]
    y_synthetic = y_resampled[original_size:]

    # Limit to requested number
    if len(X_synthetic) > n_samples:
        indices = np.random.choice(len(X_synthetic), n_samples, replace=False)
        X_synthetic = X_synthetic.iloc[indices]
        y_synthetic = y_synthetic.iloc[indices]

    return X_synthetic, y_synthetic
```

### 3. ADASYN (Adaptive Synthetic Sampling)

**Best for**: Severely imbalanced datasets

#### Features

- **Adaptive density-based sampling** for difficult examples
- **Automatic learning difficulty assessment**
- **Focused synthesis** in challenging regions
- **Improved boundary learning**

#### Implementation

```python
def adasyn_augmentation(X, y, n_samples):
    """ADASYN adaptive synthetic sampling."""
    try:
        # Configure ADASYN
        adasyn = ADASYN(
            sampling_strategy='auto',
            n_neighbors=5,
            random_state=RND
        )

        # Apply adaptive sampling
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Extract synthetic samples
        original_size = len(X)
        X_synthetic = X_resampled[original_size:]
        y_synthetic = y_resampled[original_size:]

        return X_synthetic[:n_samples], y_synthetic[:n_samples]

    except ValueError as e:
        logger.warning(f"ADASYN failed: {e}, falling back to SMOTE")
        return smote_augmentation(X, y, n_samples)
```

### 4. Basic Augmentation

**Best for**: Categorical datasets or as fallback method

#### Features

- **Simple noise addition** for numerical features
- **Category resampling** for categorical features
- **Gaussian noise injection** with adaptive variance
- **Feature-wise independent sampling**

#### Implementation

```python
def basic_augmentation(X, y, n_samples):
    """Basic augmentation with noise and resampling."""
    synthetic_samples = []
    synthetic_labels = []

    for _ in range(n_samples):
        # Random sample selection
        idx = np.random.choice(len(X))
        base_sample = X.iloc[idx].copy()
        base_label = y.iloc[idx]

        # Add controlled noise to numerical features
        for col in X.select_dtypes(include=[np.number]).columns:
            std = X[col].std()
            noise = np.random.normal(0, std * 0.1)  # 10% noise
            base_sample[col] += noise

        # Random resampling for categorical features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if np.random.random() < 0.2:  # 20% chance to change
                base_sample[col] = np.random.choice(X[col].unique())

        synthetic_samples.append(base_sample)
        synthetic_labels.append(base_label)

    X_synthetic = pd.DataFrame(synthetic_samples)
    y_synthetic = pd.Series(synthetic_labels)

    return X_synthetic, y_synthetic
```

## Quality Control Framework

### Multi-Dimensional Quality Assessment

The pipeline implements comprehensive quality control to ensure synthetic data maintains the statistical properties of the original dataset:

#### 1. Distribution Similarity

```python
def calculate_distribution_similarity(synthetic_sample, original_data):
    """Measure how well synthetic data matches original distributions."""
    similarity_scores = []

    for column in original_data.columns:
        if original_data[column].dtype in ['int64', 'float64']:
            # Numerical: KS test
            ks_stat, p_value = kstest(
                synthetic_sample[column],
                original_data[column]
            )
            similarity_scores.append(1 - ks_stat)
        else:
            # Categorical: frequency similarity
            orig_freq = original_data[column].value_counts(normalize=True)
            synth_freq = synthetic_sample[column].value_counts(normalize=True)
            similarity = 1 - np.sum(np.abs(orig_freq - synth_freq)) / 2
            similarity_scores.append(similarity)

    return np.mean(similarity_scores)
```

#### 2. Correlation Preservation

```python
def calculate_correlation_preservation(synthetic_data, original_data):
    """Measure how well correlations are preserved."""
    # Calculate correlation matrices
    orig_corr = original_data.corr()
    synth_corr = synthetic_data.corr()

    # Compute correlation difference
    corr_diff = np.abs(orig_corr - synth_corr)

    # Return preservation score (1 - mean absolute difference)
    return 1 - np.mean(corr_diff.values[~np.isnan(corr_diff.values)])
```

#### 3. Anomaly Detection

```python
def detect_anomalies(synthetic_data, original_data):
    """Detect anomalous synthetic samples."""
    # Train isolation forest on original data
    iso_forest = IsolationForest(
        contamination=0.1,
        random_state=RND
    )
    iso_forest.fit(original_data)

    # Score synthetic samples
    anomaly_scores = iso_forest.decision_function(synthetic_data)

    # Convert to quality scores (higher is better)
    quality_scores = (anomaly_scores - anomaly_scores.min()) / (
        anomaly_scores.max() - anomaly_scores.min()
    )

    return quality_scores
```

### Diversity Control

#### Diversity Calculation

```python
def calculate_diversity_score(data):
    """Calculate dataset diversity using pairwise distances."""
    # Sample subset for computational efficiency
    sample_size = min(1000, len(data))
    sample_data = data.sample(n=sample_size, random_state=RND)

    # Calculate pairwise distances
    distances = pdist(sample_data, metric='euclidean')

    # Normalize by maximum possible distance
    max_distance = np.sqrt(sample_data.shape[1])
    normalized_distances = distances / max_distance

    # Diversity is mean pairwise distance
    diversity = np.mean(normalized_distances)

    return diversity
```

#### Quality Filtering Pipeline

```python
def enhanced_quality_filtering(synthetic_samples, original_samples):
    """Comprehensive quality filtering pipeline."""
    quality_scores = []

    for idx, sample in synthetic_samples.iterrows():
        # Multiple quality metrics
        dist_score = calculate_distribution_similarity(
            pd.DataFrame([sample]), original_samples
        )
        corr_score = calculate_correlation_preservation(
            pd.DataFrame([sample]), original_samples
        )
        anomaly_score = 1 - isolation_forest.decision_function([sample])[0]

        # Weighted combined score
        combined_score = (
            0.4 * dist_score +
            0.3 * corr_score +
            0.3 * anomaly_score
        )
        quality_scores.append(combined_score)

    # Filter by quality threshold
    quality_mask = np.array(quality_scores) >= QUALITY_THRESHOLD
    filtered_samples = synthetic_samples[quality_mask]

    # Check diversity
    diversity_score = calculate_diversity_score(filtered_samples)

    logger.info(f"Quality filtering: {len(filtered_samples)}/{len(synthetic_samples)} samples kept")
    logger.info(f"Avg quality score: {np.mean(np.array(quality_scores)[quality_mask]):.3f}")
    logger.info(f"Diversity score: {diversity_score:.3f}")

    return filtered_samples, quality_scores
```

## Configuration Options

### Augmentation Parameters

```python
# Enable/disable augmentation
ENABLE_DATA_AUGMENTATION = True

# Method selection
AUGMENTATION_METHOD = "auto"  # "auto", "sdv_copula", "smote", "adasyn", "basic"

# Augmentation amount
AUGMENTATION_RATIO = 0.05  # 5% of original dataset size

# Quality control thresholds
QUALITY_THRESHOLD = 0.7    # Minimum quality score (0-1)
DIVERSITY_THRESHOLD = 0.95 # Minimum diversity score (0-1)

# Method-specific parameters
SDV_EPOCHS = 100           # Training epochs for SDV (5 in testing mode)
SMOTE_K_NEIGHBORS = 5      # k for SMOTE (auto-adjusted)
BASIC_NOISE_FACTOR = 0.1   # Noise factor for basic method
```

### Adaptive Ratio Calculation

```python
def calculate_adaptive_ratio(data_characteristics):
    """Calculate optimal augmentation ratio based on data properties."""
    base_ratio = 0.05  # 5% baseline

    # Adjust based on dataset size
    if data_characteristics['is_small_dataset']:
        ratio_multiplier = 2.0  # More augmentation for small datasets
    else:
        ratio_multiplier = 1.0

    # Adjust based on class imbalance
    if data_characteristics['is_imbalanced']:
        ratio_multiplier *= 1.5  # More for imbalanced data

    # Adjust based on complexity
    complexity = data_characteristics['feature_complexity']
    if complexity > 0.8:
        ratio_multiplier *= 0.8  # Less for very complex data

    final_ratio = base_ratio * ratio_multiplier
    return min(final_ratio, 0.2)  # Cap at 20%
```

## Performance Optimization

### Memory Management

- **Batch processing** for large synthetic datasets
- **Streaming generation** to reduce memory footprint
- **Garbage collection** after augmentation
- **Memory monitoring** with warnings

### Computational Efficiency

- **Parallel processing** where applicable
- **Early stopping** for quality control
- **Caching** of expensive computations
- **Progress tracking** for long operations

### GPU Acceleration (Future)

- **CUDA support** for neural-based methods
- **GPU-accelerated SDV** operations
- **Parallel quality assessment**

## Best Practices

### When to Use Augmentation

✅ **Recommended**:

- Small to medium datasets (<10K samples)
- Class imbalanced problems
- High-stakes applications requiring robustness
- When overfitting is detected

❌ **Not Recommended**:

- Very large datasets (>100K samples)
- When computational resources are limited
- For quick prototyping/testing
- When original data quality is poor

### Parameter Tuning Guidelines

#### Augmentation Ratio

- **Start small**: Begin with 5% and increase gradually
- **Monitor performance**: Watch for diminishing returns
- **Consider data size**: Smaller datasets need higher ratios
- **Validate carefully**: Check for data leakage

#### Quality Thresholds

- **Conservative start**: Begin with 0.8 threshold
- **Adjust based on results**: Lower if too few samples pass
- **Domain-specific tuning**: Some domains need stricter quality
- **Cross-validate**: Ensure augmented data improves CV scores

### Debugging Augmentation

#### Common Issues

1. **Poor quality synthetic data**
   - Check feature distributions
   - Adjust quality thresholds
   - Try different methods

2. **Performance degradation**
   - Reduce augmentation ratio
   - Increase quality filtering
   - Validate data integrity

3. **Memory issues**
   - Reduce batch sizes
   - Use streaming generation
   - Monitor memory usage

#### Diagnostic Tools

```python
def diagnose_augmentation_quality(original, synthetic):
    """Comprehensive augmentation quality diagnosis."""
    report = {
        'statistical_tests': perform_statistical_tests(original, synthetic),
        'distribution_plots': create_distribution_plots(original, synthetic),
        'correlation_analysis': analyze_correlation_preservation(original, synthetic),
        'anomaly_detection': detect_synthetic_anomalies(original, synthetic),
        'performance_impact': evaluate_model_performance_impact(original, synthetic)
    }
    return report
```

## Future Enhancements

### Planned Features

- **GAN-based synthesis** for complex data patterns
- **Variational Autoencoders** for latent space sampling
- **Conditional generation** for targeted augmentation
- **Active learning integration** for selective augmentation

### Research Directions

- **Meta-learning** for automatic method selection
- **Quality-aware training** for synthesis models
- **Federated augmentation** for privacy-preserving synthesis
- **Domain adaptation** for cross-domain augmentation

---

_This guide covers the current augmentation capabilities. For the latest features, check the repository and technical documentation._
