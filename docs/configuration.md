# Configuration Guide

## Overview

The Six-Stack Personality Classification Pipeline provides extensive configuration options through the centralized `src/modules/config.py` file. This guide covers all configuration parameters, their purposes, and best practices for tuning.

## Configuration Architecture

### Centralized Configuration

All configuration is managed through a single module to ensure:
- **Consistency** across all components
- **Easy maintenance** and updates
- **Environment-specific** settings
- **Type safety** with enums and validation

### Configuration Categories

1. **Core Parameters** - Basic pipeline settings
2. **Threading Configuration** - Parallel processing control
3. **Data Augmentation** - Synthetic data generation
4. **Model Training** - Algorithm-specific settings
5. **Development** - Testing and debugging options
6. **Logging** - Output and monitoring control

## Core Parameters

### Reproducibility Settings
```python
# Global random seed for reproducibility
RND: int = 42

# Description: Controls all random number generation across the pipeline
# Impact: Ensures reproducible results across runs
# Tuning: Change only when you need different random behavior
# Valid Range: Any integer (0-2^31)
```

### Cross-Validation Configuration
```python
# Number of stratified folds for cross-validation
N_SPLITS: int = 5

# Description: Controls k-fold cross-validation splitting
# Impact: More folds = more reliable estimates but longer training
# Tuning: 3-10 folds typically, 5 is standard
# Memory Impact: Linear increase with more folds
```

### Hyperparameter Optimization
```python
# Optuna trials per individual stack
N_TRIALS_STACK: int = 15

# Description: Number of hyperparameter combinations to try per stack
# Impact: More trials = better optimization but longer training
# Tuning Guidelines:
#   - Development: 5-15 trials
#   - Production: 50-200 trials
#   - Competition: 500+ trials
# Time Impact: Linear increase with trial count

# Ensemble blending optimization trials
N_TRIALS_BLEND: int = 200

# Description: Trials for optimizing ensemble weights
# Impact: Critical for final performance, usually converges quickly
# Tuning: 100-500 trials, diminishing returns after 200
```

## Threading Configuration

### Thread Management Enum
```python
class ThreadConfig(Enum):
    """Centralized threading configuration for all models."""
    
    N_JOBS: int = 4          # sklearn parallel jobs
    THREAD_COUNT: int = 4    # XGBoost/LightGBM threads
```

### Optimization Guidelines

#### System-Specific Tuning
```python
# For development machines (4-8 cores)
N_JOBS = 2
THREAD_COUNT = 2

# For production servers (16+ cores)
N_JOBS = 8
THREAD_COUNT = 8

# For memory-constrained environments
N_JOBS = 1
THREAD_COUNT = 1

# Auto-detection approach
import multiprocessing
optimal_threads = min(multiprocessing.cpu_count(), 8)
```

#### Performance vs Resource Trade-offs

| Setting | Training Speed | Memory Usage | CPU Usage |
|---------|---------------|--------------|-----------|
| 1 thread | Slowest | Lowest | Low |
| 2-4 threads | Moderate | Moderate | Medium |
| 8+ threads | Fastest | Highest | High |

## Data Augmentation Configuration

### Main Augmentation Settings
```python
# Enable/disable data augmentation globally
ENABLE_DATA_AUGMENTATION: bool = True

# Augmentation method selection
AUGMENTATION_METHOD: str = "sdv_copula"
# Options: "auto", "sdv_copula", "smote", "adasyn", "basic"

# Augmentation ratio (fraction of original dataset)
AUGMENTATION_RATIO: float = 0.05  # 5% additional synthetic data
```

### Method Selection Guide

#### "auto" (Recommended)
- **Best for**: Most use cases
- **Behavior**: Automatically selects optimal method based on data characteristics
- **Fallback**: Always provides a working solution

#### "sdv_copula"
- **Best for**: Large datasets with complex distributions
- **Pros**: High-quality synthetic data, preserves correlations
- **Cons**: Computationally intensive, requires more memory
- **Use when**: Dataset >5K samples, complex feature interactions

#### "smote"
- **Best for**: Small to medium datasets with class imbalance
- **Pros**: Fast, well-tested, handles imbalance well
- **Cons**: May create unrealistic edge cases
- **Use when**: Dataset <5K samples, clear class imbalance

#### "adasyn"
- **Best for**: Severely imbalanced datasets
- **Pros**: Adaptive to difficult examples, improved boundary learning
- **Cons**: Sensitive to noise, may overfit to outliers
- **Use when**: Extreme imbalance (>90% majority class)

#### "basic"
- **Best for**: High-categorical datasets or fallback
- **Pros**: Fast, simple, always works
- **Cons**: Lower quality, limited sophistication
- **Use when**: Many categorical features, quick prototyping

### Quality Control Parameters
```python
# Quality filtering threshold (0-1, higher = stricter)
QUALITY_THRESHOLD: float = 0.7

# Diversity requirement (0-1, higher = more diverse)
DIVERSITY_THRESHOLD: float = 0.95

# Method-specific parameters
SDV_EPOCHS: int = 100                # SDV training epochs (5 in testing)
SMOTE_K_NEIGHBORS: int = 5           # k for SMOTE (auto-adjusted)
BASIC_NOISE_FACTOR: float = 0.1      # Noise factor for basic method
```

### Advanced Augmentation Tuning

#### Quality Threshold Tuning
```python
# Conservative (high quality, fewer samples)
QUALITY_THRESHOLD = 0.8

# Balanced (moderate quality, moderate samples)
QUALITY_THRESHOLD = 0.7

# Aggressive (lower quality, more samples)
QUALITY_THRESHOLD = 0.6

# Development/testing (relaxed quality)
QUALITY_THRESHOLD = 0.5
```

#### Ratio Optimization Strategy
```python
# Start conservative and increase
AUGMENTATION_RATIOS = [0.02, 0.05, 0.10, 0.15, 0.20]

# Monitor cross-validation scores
for ratio in AUGMENTATION_RATIOS:
    AUGMENTATION_RATIO = ratio
    cv_score = evaluate_pipeline()
    if cv_score < previous_best:
        break  # Diminishing returns detected
```

## Model Training Configuration

### Label Noise for Robustness
```python
# Label noise rate for Stack F (noise-robust training)
LABEL_NOISE_RATE: float = 0.02  # 2% of labels randomly flipped

# Description: Improves generalization by training on noisy labels
# Impact: Better robustness to annotation errors
# Tuning Range: 0.01-0.05 (1-5%)
# Warning: Too much noise degrades performance
```

### Timeout and Resource Limits
```python
# Training timeout per stack (seconds)
STACK_TIMEOUT: int = 1800  # 30 minutes

# Memory limit warning threshold (GB)
MEMORY_WARNING_THRESHOLD: float = 8.0

# Early stopping patience for neural networks
EARLY_STOPPING_PATIENCE: int = 10
```

## Development and Testing

### Testing Mode Configuration
```python
# Enable reduced dataset for faster development
TESTING_MODE: bool = True

# Sample size in testing mode
TESTING_SAMPLE_SIZE: int = 1000

# Reduced trials in testing mode
TESTING_N_TRIALS_STACK: int = 5
TESTING_N_TRIALS_BLEND: int = 50

# Fast augmentation in testing
TESTING_SDV_EPOCHS: int = 5
```

### Development Presets
```python
# Quick development preset
def configure_for_development():
    global TESTING_MODE, N_TRIALS_STACK, ENABLE_DATA_AUGMENTATION
    TESTING_MODE = True
    N_TRIALS_STACK = 5
    ENABLE_DATA_AUGMENTATION = False
    logger.info("Configured for rapid development")

# Full production preset
def configure_for_production():
    global TESTING_MODE, N_TRIALS_STACK, N_TRIALS_BLEND
    TESTING_MODE = False
    N_TRIALS_STACK = 100
    N_TRIALS_BLEND = 300
    logger.info("Configured for production run")
```

## Logging Configuration

### Log Level Settings
```python
# Logging level
LOG_LEVEL: str = "INFO"

# Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
# DEBUG: Very detailed information for debugging
# INFO: General information about progress (recommended)
# WARNING: Important warnings and issues
# ERROR: Only error messages
```

### Advanced Logging Configuration
```python
# Log file configuration
LOG_FILE: str = "personality_classifier.log"
LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT: int = 5

# Performance logging
ENABLE_PERFORMANCE_LOGGING: bool = True
LOG_MEMORY_USAGE: bool = True
LOG_TIMING_INFO: bool = True

# Progress bar configuration
SHOW_PROGRESS_BARS: bool = True
PROGRESS_BAR_STYLE: str = "tqdm"  # "tqdm" or "simple"
```

## Environment-Specific Configuration

### Configuration Profiles

#### Local Development
```python
# config_development.py
TESTING_MODE = True
N_TRIALS_STACK = 5
N_TRIALS_BLEND = 50
ENABLE_DATA_AUGMENTATION = False
LOG_LEVEL = "DEBUG"
ThreadConfig.N_JOBS = 2
ThreadConfig.THREAD_COUNT = 2
```

#### CI/CD Pipeline
```python
# config_ci.py
TESTING_MODE = True
N_TRIALS_STACK = 3
N_TRIALS_BLEND = 20
ENABLE_DATA_AUGMENTATION = False
LOG_LEVEL = "WARNING"
ThreadConfig.N_JOBS = 1
ThreadConfig.THREAD_COUNT = 1
```

#### Production Server
```python
# config_production.py
TESTING_MODE = False
N_TRIALS_STACK = 100
N_TRIALS_BLEND = 300
ENABLE_DATA_AUGMENTATION = True
AUGMENTATION_METHOD = "sdv_copula"
LOG_LEVEL = "INFO"
ThreadConfig.N_JOBS = 8
ThreadConfig.THREAD_COUNT = 8
```

### Environment Variable Integration
```python
import os

# Override with environment variables
N_TRIALS_STACK = int(os.getenv('PERSONALITY_TRIALS_STACK', N_TRIALS_STACK))
TESTING_MODE = os.getenv('PERSONALITY_TESTING_MODE', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('PERSONALITY_LOG_LEVEL', LOG_LEVEL)

# Docker environment detection
if os.getenv('RUNNING_IN_DOCKER'):
    ThreadConfig.N_JOBS = min(ThreadConfig.N_JOBS, 4)
    MEMORY_WARNING_THRESHOLD = 2.0  # Lower threshold in containers
```

## Performance Tuning Guidelines

### Memory Optimization
```python
# For systems with <8GB RAM
TESTING_MODE = True
TESTING_SAMPLE_SIZE = 500
ThreadConfig.N_JOBS = 1
ENABLE_DATA_AUGMENTATION = False

# For systems with 8-16GB RAM (recommended)
TESTING_SAMPLE_SIZE = 1000
ThreadConfig.N_JOBS = 2
AUGMENTATION_RATIO = 0.03

# For systems with >16GB RAM
ThreadConfig.N_JOBS = 4
AUGMENTATION_RATIO = 0.05
N_TRIALS_STACK = 50
```

### Speed Optimization
```python
# Fastest configuration (for quick iteration)
TESTING_MODE = True
N_TRIALS_STACK = 3
N_TRIALS_BLEND = 20
ENABLE_DATA_AUGMENTATION = False
SHOW_PROGRESS_BARS = False

# Balanced configuration (development)
N_TRIALS_STACK = 15
N_TRIALS_BLEND = 100
AUGMENTATION_METHOD = "smote"  # Faster than SDV

# Quality-focused configuration (production)
N_TRIALS_STACK = 100
N_TRIALS_BLEND = 300
AUGMENTATION_METHOD = "sdv_copula"
```

### GPU Configuration (Future)
```python
# GPU settings (when available)
USE_GPU: bool = False
GPU_MEMORY_FRACTION: float = 0.8
ENABLE_MIXED_PRECISION: bool = False

# GPU-specific model settings
GPU_BATCH_SIZE: int = 64
GPU_N_ESTIMATORS_FACTOR: float = 2.0  # Increase for GPU
```

## Validation and Error Handling

### Configuration Validation
```python
def validate_configuration():
    """Validate configuration parameters."""
    assert 0 < AUGMENTATION_RATIO <= 1.0, "Invalid augmentation ratio"
    assert N_SPLITS >= 2, "Need at least 2 CV folds"
    assert 0 <= LABEL_NOISE_RATE <= 0.2, "Label noise rate too high"
    assert ThreadConfig.N_JOBS >= 1, "Need at least 1 job"
    
    if TESTING_MODE and N_TRIALS_STACK > 20:
        logger.warning("High trial count in testing mode may be slow")
    
    if not ENABLE_DATA_AUGMENTATION and AUGMENTATION_RATIO > 0:
        logger.warning("Augmentation ratio set but augmentation disabled")
```

### Configuration Debugging
```python
def log_configuration():
    """Log current configuration for debugging."""
    logger.info("Configuration Summary:")
    logger.info(f"  Mode: {'Testing' if TESTING_MODE else 'Production'}")
    logger.info(f"  Trials per stack: {N_TRIALS_STACK}")
    logger.info(f"  Augmentation: {AUGMENTATION_METHOD if ENABLE_DATA_AUGMENTATION else 'Disabled'}")
    logger.info(f"  Threading: {ThreadConfig.N_JOBS} jobs, {ThreadConfig.THREAD_COUNT} threads")
    logger.info(f"  Random seed: {RND}")
```

## Configuration Best Practices

### 1. Start Conservative
- Begin with default settings
- Use testing mode for development
- Gradually increase complexity

### 2. Monitor Resources
- Watch memory usage during training
- Monitor CPU utilization
- Adjust threading based on available resources

### 3. Validate Changes
- Test configuration changes on small datasets first
- Compare cross-validation scores
- Ensure reproducibility with fixed seeds

### 4. Document Customizations
- Comment configuration changes
- Track performance impacts
- Maintain environment-specific configs

### 5. Use Version Control
- Track configuration changes
- Tag configurations with results
- Maintain separate configs for different environments

## Troubleshooting Common Issues

### Memory Issues
```python
# Reduce memory usage
TESTING_MODE = True
ThreadConfig.N_JOBS = 1
ENABLE_DATA_AUGMENTATION = False
TESTING_SAMPLE_SIZE = 500
```

### Slow Training
```python
# Speed up training
N_TRIALS_STACK = 5
N_TRIALS_BLEND = 50
AUGMENTATION_METHOD = "basic"
SHOW_PROGRESS_BARS = False
```

### Poor Performance
```python
# Increase optimization
N_TRIALS_STACK = 100
N_TRIALS_BLEND = 300
AUGMENTATION_METHOD = "sdv_copula"
AUGMENTATION_RATIO = 0.08
```

### Reproducibility Issues
```python
# Ensure reproducibility
# Set fixed seed
RND = 42

# Single-threaded for determinism
ThreadConfig.N_JOBS = 1
ThreadConfig.THREAD_COUNT = 1

# Disable random augmentation
ENABLE_DATA_AUGMENTATION = False
```

---

*This configuration guide covers all current options. For the latest parameters and features, check the source code in `src/modules/config.py`.*
