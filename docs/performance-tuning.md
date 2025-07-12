# Performance Tuning Guide

## Overview

This guide provides comprehensive strategies for optimizing the Six-Stack Personality Classification Pipeline performance across different dimensions: speed, memory usage, accuracy, and resource utilization.

## Performance Dimensions

### 1. Training Speed
- Hyperparameter optimization trials
- Data augmentation complexity
- Threading configuration
- Model complexity

### 2. Memory Efficiency  
- Dataset size management
- Model memory footprint
- Parallel processing overhead
- Synthetic data generation

### 3. Prediction Accuracy
- Ensemble optimization
- Cross-validation strategy
- Feature engineering
- Model diversity

### 4. Resource Utilization
- CPU core usage
- Memory allocation
- I/O optimization
- Caching strategies

## Speed Optimization

### Quick Development Setup
```python
# Ultra-fast configuration for development iteration
TESTING_MODE = True
TESTING_SAMPLE_SIZE = 500
N_TRIALS_STACK = 3
N_TRIALS_BLEND = 20
ENABLE_DATA_AUGMENTATION = False
SHOW_PROGRESS_BARS = False

# Expected time: 2-3 minutes
# Accuracy trade-off: 2-3% lower than production
```

### Balanced Development Setup
```python
# Moderate speed with reasonable accuracy
TESTING_MODE = True
TESTING_SAMPLE_SIZE = 1000
N_TRIALS_STACK = 10
N_TRIALS_BLEND = 50
AUGMENTATION_METHOD = "smote"  # Faster than SDV
AUGMENTATION_RATIO = 0.03

# Expected time: 5-8 minutes
# Accuracy trade-off: 1-2% lower than production
```

### Production Speed Optimization
```python
# Optimize for fastest production runs
N_TRIALS_STACK = 50          # Reduced from 100
N_TRIALS_BLEND = 150         # Reduced from 300
AUGMENTATION_METHOD = "smote" # Faster than SDV Copula
ThreadConfig.N_JOBS = max_cores
ThreadConfig.THREAD_COUNT = max_cores

# Expected time: 15-20 minutes (vs 30-45 full)
# Accuracy trade-off: 0.5-1% lower
```

### Algorithm-Specific Speed Tuning

#### XGBoost Optimization
```python
def optimize_xgboost_speed(trial):
    return {
        'n_estimators': trial.suggest_int('xgb_n', 100, 500),  # Reduced range
        'max_depth': trial.suggest_int('xgb_d', 3, 6),         # Shallower trees
        'learning_rate': trial.suggest_float('xgb_lr', 0.05, 0.3),  # Higher LR
        'tree_method': 'hist',          # Faster algorithm
        'grow_policy': 'lossguide',     # More efficient growth
        'nthread': ThreadConfig.THREAD_COUNT,
    }
```

#### LightGBM Optimization
```python
def optimize_lightgbm_speed(trial):
    return {
        'n_estimators': trial.suggest_int('lgb_n', 100, 400),
        'max_depth': trial.suggest_int('lgb_d', 3, 5),
        'learning_rate': trial.suggest_float('lgb_lr', 0.05, 0.2),
        'feature_fraction': 0.8,        # Use subset of features
        'bagging_fraction': 0.8,        # Use subset of samples
        'bagging_freq': 5,              # Enable bagging
        'num_threads': ThreadConfig.THREAD_COUNT,
    }
```

#### Neural Network Speed Tuning
```python
def optimize_neural_network_speed(trial):
    return {
        'hidden_layer_sizes': [(64,), (128,), (64, 32)],  # Smaller networks
        'max_iter': 200,                # Reduced iterations
        'early_stopping': True,         # Stop when converged
        'validation_fraction': 0.1,     # Small validation set
        'n_iter_no_change': 10,        # Early stopping patience
        'learning_rate_init': 0.001,    # Good starting LR
    }
```

## Memory Optimization

### Memory-Constrained Environments (<4GB)
```python
# Minimal memory configuration
TESTING_MODE = True
TESTING_SAMPLE_SIZE = 300
ThreadConfig.N_JOBS = 1
ThreadConfig.THREAD_COUNT = 1
ENABLE_DATA_AUGMENTATION = False
N_TRIALS_STACK = 3

# Memory usage: ~1-2GB peak
```

### Standard Environments (4-8GB)
```python
# Balanced memory usage
TESTING_SAMPLE_SIZE = 1000
ThreadConfig.N_JOBS = 2
AUGMENTATION_RATIO = 0.03
N_TRIALS_STACK = 15

# Memory usage: ~3-6GB peak
```

### High-Memory Environments (16GB+)
```python
# Full dataset with aggressive augmentation
TESTING_MODE = False
ThreadConfig.N_JOBS = 4
AUGMENTATION_RATIO = 0.08
N_TRIALS_STACK = 100

# Memory usage: ~8-12GB peak
```

### Memory Usage Monitoring
```python
import psutil
import gc

def monitor_memory_usage():
    """Monitor and log memory usage during training."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024**3:.1f} GB")
    
    if memory_info.rss > MEMORY_WARNING_THRESHOLD * 1024**3:
        logger.warning("High memory usage detected")
        gc.collect()  # Force garbage collection

def optimize_memory_for_augmentation():
    """Optimize memory usage during data augmentation."""
    # Process in batches
    batch_size = min(1000, TESTING_SAMPLE_SIZE // 4)
    
    # Clear intermediate variables
    gc.collect()
    
    # Use memory-efficient data types
    X = X.astype('float32')  # Use float32 instead of float64
```

### Memory-Efficient Data Structures
```python
def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage."""
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'object':
            if df[col].nunique() < 50:  # Convert to category
                df[col] = df[col].astype('category')
    
    return df
```

## Accuracy Optimization

### Hyperparameter Search Intensity
```python
# Conservative search (baseline)
N_TRIALS_STACK = 50
N_TRIALS_BLEND = 200

# Intensive search (competition-grade)
N_TRIALS_STACK = 200
N_TRIALS_BLEND = 500

# Extreme search (research-grade)
N_TRIALS_STACK = 1000
N_TRIALS_BLEND = 1000
```

### Advanced Ensemble Strategies
```python
def advanced_ensemble_optimization():
    """Advanced ensemble optimization strategies."""
    
    # Multi-level blending
    def create_meta_ensemble(oof_predictions):
        # Level 1: Blend similar stacks
        traditional_blend = blend_stacks(['A', 'B'], oof_predictions)
        boosting_blend = blend_stacks(['C'], oof_predictions)
        diverse_blend = blend_stacks(['D', 'E'], oof_predictions)
        robust_blend = blend_stacks(['F'], oof_predictions)
        
        # Level 2: Meta-blend the blends
        meta_predictions = [traditional_blend, boosting_blend, 
                          diverse_blend, robust_blend]
        return optimize_meta_weights(meta_predictions)
    
    # Dynamic weight adjustment
    def dynamic_ensemble_weights(validation_scores):
        """Adjust ensemble weights based on validation performance."""
        weights = {}
        total_score = sum(validation_scores.values())
        
        for stack, score in validation_scores.items():
            # Higher performing stacks get higher weights
            weights[stack] = score / total_score
        
        return weights
```

### Cross-Validation Strategy Optimization
```python
# Standard stratified K-fold
N_SPLITS = 5

# More robust (slower but more reliable)
N_SPLITS = 10

# Nested CV for unbiased estimation
def nested_cross_validation(X, y):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RND)
    
    nested_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        # Inner loop for hyperparameter optimization
        best_model = optimize_on_fold(X[train_idx], y[train_idx], inner_cv)
        
        # Outer loop for unbiased evaluation
        score = best_model.score(X[test_idx], y[test_idx])
        nested_scores.append(score)
    
    return np.mean(nested_scores)
```

### Feature Engineering Optimization
```python
def advanced_feature_engineering(df):
    """Advanced feature engineering for better accuracy."""
    
    # Polynomial features (degree 2)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    poly_features = poly.fit_transform(df.select_dtypes(include=[np.number]))
    
    # Feature selection based on importance
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=20)
    selected_features = selector.fit_transform(poly_features, y)
    
    # Target encoding for categorical features
    for col in df.select_dtypes(include=['object']).columns:
        target_mean = df.groupby(col)['target'].mean()
        df[f'{col}_target_encoded'] = df[col].map(target_mean)
    
    return df
```

## Threading and Parallelization

### Optimal Threading Configuration

#### CPU Core Detection
```python
import multiprocessing
import psutil

def detect_optimal_threading():
    """Automatically detect optimal threading configuration."""
    
    # Physical cores (exclude hyperthreading)
    physical_cores = psutil.cpu_count(logical=False)
    
    # Logical cores (include hyperthreading)
    logical_cores = multiprocessing.cpu_count()
    
    # Available memory per core
    memory_gb = psutil.virtual_memory().total / (1024**3)
    memory_per_core = memory_gb / logical_cores
    
    # Optimal configuration based on system
    if memory_per_core < 2:  # Memory constrained
        recommended_threads = max(1, physical_cores // 2)
    elif memory_per_core > 4:  # Memory abundant
        recommended_threads = logical_cores
    else:  # Balanced
        recommended_threads = physical_cores
    
    return min(recommended_threads, 8)  # Cap at 8 for stability
```

#### Threading Strategy by Component
```python
# Different threading for different components
class ComponentThreading:
    # Data preprocessing (I/O bound)
    PREPROCESSING_THREADS = max(2, multiprocessing.cpu_count() // 2)
    
    # Model training (CPU bound)
    TRAINING_THREADS = min(4, multiprocessing.cpu_count())
    
    # Ensemble generation (memory bound)
    ENSEMBLE_THREADS = min(2, multiprocessing.cpu_count())
    
    # Hyperparameter optimization (parallel trials)
    OPTUNA_THREADS = 1  # Sequential for stability
```

### Parallel Processing Strategies

#### Parallel Stack Training
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_stack_training(stack_configs, data):
    """Train multiple stacks in parallel."""
    
    # Limit concurrent processes to avoid memory issues
    max_workers = min(3, mp.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all stack training jobs
        futures = {
            executor.submit(train_single_stack, config, data): config.name
            for config in stack_configs
        }
        
        # Collect results as they complete
        studies = {}
        for future in concurrent.futures.as_completed(futures):
            stack_name = futures[future]
            studies[stack_name] = future.result()
    
    return studies
```

#### Memory-Aware Parallel Processing
```python
def memory_aware_parallel_processing(tasks, data_size_gb):
    """Adjust parallelism based on memory requirements."""
    
    available_memory = psutil.virtual_memory().available / (1024**3)
    memory_per_task = data_size_gb * 2  # Rough estimate
    
    max_parallel = int(available_memory // memory_per_task)
    max_parallel = max(1, min(max_parallel, mp.cpu_count()))
    
    logger.info(f"Running {len(tasks)} tasks with {max_parallel} parallel workers")
    
    return max_parallel
```

## I/O and Data Pipeline Optimization

### Data Loading Optimization
```python
def optimized_data_loading():
    """Optimize data loading for better performance."""
    
    # Use efficient file formats
    # CSV -> Parquet for better compression and speed
    def save_as_parquet(df, filename):
        df.to_parquet(f"{filename}.parquet", compression='snappy')
    
    def load_from_parquet(filename):
        return pd.read_parquet(f"{filename}.parquet")
    
    # Lazy loading for large datasets
    def lazy_data_loader(file_paths, chunk_size=10000):
        for file_path in file_paths:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
    
    # Memory mapping for repeated access
    def create_memory_mapped_data(X, y):
        X_mmap = np.memmap('X_temp.dat', dtype='float32', mode='w+', shape=X.shape)
        y_mmap = np.memmap('y_temp.dat', dtype='int32', mode='w+', shape=y.shape)
        
        X_mmap[:] = X[:]
        y_mmap[:] = y[:]
        
        return X_mmap, y_mmap
```

### Caching Strategies
```python
import joblib
from functools import lru_cache

# Cache expensive computations
@lru_cache(maxsize=128)
def cached_feature_engineering(data_hash):
    """Cache feature engineering results."""
    # Expensive feature engineering here
    pass

# Persistent caching
def cache_preprocessing_results(X, y, cache_dir='./cache'):
    """Cache preprocessing results to disk."""
    cache_key = hashlib.md5(str(X.values).encode()).hexdigest()[:8]
    cache_file = f"{cache_dir}/preprocessing_{cache_key}.pkl"
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached preprocessing from {cache_file}")
        return joblib.load(cache_file)
    else:
        # Perform preprocessing
        result = expensive_preprocessing(X, y)
        
        # Cache for future use
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(result, cache_file)
        logger.info(f"Cached preprocessing to {cache_file}")
        
        return result
```

## Model-Specific Optimizations

### XGBoost Performance Tuning
```python
def optimize_xgboost_performance():
    """XGBoost-specific performance optimizations."""
    
    return {
        # Use histogram-based algorithm
        'tree_method': 'hist',
        
        # Enable GPU if available
        'tree_method': 'gpu_hist' if gpu_available else 'hist',
        
        # Optimize memory usage
        'max_bin': 256,  # Reduce memory usage
        'grow_policy': 'lossguide',  # More memory efficient
        
        # Threading
        'nthread': ThreadConfig.THREAD_COUNT,
        
        # Caching
        'predictor': 'cpu_predictor',  # Faster prediction
        
        # Early stopping
        'early_stopping_rounds': 10,
    }
```

### LightGBM Performance Tuning
```python
def optimize_lightgbm_performance():
    """LightGBM-specific performance optimizations."""
    
    return {
        # Optimize speed
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        
        # Memory optimization
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        
        # Threading
        'num_threads': ThreadConfig.THREAD_COUNT,
        
        # Speed optimizations
        'force_row_wise': True,  # Better for small datasets
        'histogram_pool_size': 8192,  # Optimize memory
        
        # Early stopping
        'early_stopping_round': 10,
    }
```

### CatBoost Performance Tuning
```python
def optimize_catboost_performance():
    """CatBoost-specific performance optimizations."""
    
    return {
        # Speed optimizations
        'task_type': 'CPU',  # or 'GPU' if available
        'bootstrap_type': 'Bernoulli',  # Faster than Bayesian
        
        # Memory optimization
        'max_ctr_complexity': 2,  # Reduce memory for categorical features
        
        # Threading
        'thread_count': ThreadConfig.THREAD_COUNT,
        
        # Training speed
        'leaf_estimation_iterations': 1,  # Faster training
        'grow_policy': 'Depthwise',  # More predictable
        
        # Early stopping
        'early_stopping_rounds': 10,
        'use_best_model': True,
    }
```

## Monitoring and Profiling

### Performance Monitoring
```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name):
    """Monitor performance of operations."""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        duration = end_time - start_time
        memory_delta = (end_memory - start_memory) / 1024**2  # MB
        
        logger.info(f"Completed {operation_name}:")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Memory delta: {memory_delta:.1f} MB")

# Usage
with performance_monitor("Stack A Training"):
    train_stack_a()
```

### Bottleneck Detection
```python
import cProfile
import pstats

def profile_pipeline():
    """Profile the pipeline to identify bottlenecks."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run pipeline
    main()
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    # Save detailed profile
    stats.dump_stats('pipeline_profile.prof')
```

### Real-time Performance Dashboard
```python
def create_performance_dashboard():
    """Create real-time performance monitoring."""
    
    metrics = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_io': psutil.disk_io_counters(),
        'network_io': psutil.net_io_counters(),
    }
    
    # Log metrics periodically
    logger.info(f"Performance metrics: {metrics}")
    
    return metrics
```

## Environment-Specific Optimizations

### Docker Container Optimization
```python
# Docker-specific configuration
if os.getenv('RUNNING_IN_DOCKER'):
    # Reduce threading in containers
    ThreadConfig.N_JOBS = min(ThreadConfig.N_JOBS, 2)
    ThreadConfig.THREAD_COUNT = min(ThreadConfig.THREAD_COUNT, 2)
    
    # Lower memory thresholds
    MEMORY_WARNING_THRESHOLD = 2.0  # GB
    
    # Reduce dataset size
    TESTING_SAMPLE_SIZE = min(TESTING_SAMPLE_SIZE, 800)
```

### Cloud Platform Optimization

#### AWS EC2
```python
def optimize_for_aws_ec2():
    """Optimize for AWS EC2 instances."""
    
    # Detect instance type
    instance_type = get_ec2_instance_type()
    
    if 'xlarge' in instance_type:
        ThreadConfig.N_JOBS = 8
        N_TRIALS_STACK = 100
    elif 'large' in instance_type:
        ThreadConfig.N_JOBS = 4
        N_TRIALS_STACK = 50
    else:  # medium or small
        ThreadConfig.N_JOBS = 2
        N_TRIALS_STACK = 25
```

#### Google Colab
```python
def optimize_for_colab():
    """Optimize for Google Colab environment."""
    
    # Colab-specific settings
    ThreadConfig.N_JOBS = 2  # Colab has limited cores
    TESTING_SAMPLE_SIZE = 1500  # Good balance for Colab
    
    # Use GPU if available
    if torch.cuda.is_available():
        logger.info("GPU detected in Colab, enabling GPU optimizations")
        # GPU-specific configurations
```

## Performance Testing Framework

### Benchmarking Suite
```python
def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    
    configs = [
        ('Ultra Fast', ultra_fast_config()),
        ('Balanced', balanced_config()),
        ('High Quality', high_quality_config()),
    ]
    
    results = {}
    
    for config_name, config in configs:
        logger.info(f"Benchmarking {config_name} configuration")
        
        start_time = time.time()
        apply_config(config)
        
        # Run pipeline
        accuracy = run_pipeline_with_config(config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        results[config_name] = {
            'accuracy': accuracy,
            'duration': duration,
            'accuracy_per_minute': accuracy / (duration / 60)
        }
    
    # Generate benchmark report
    generate_benchmark_report(results)
    
    return results
```

---

*This performance tuning guide provides comprehensive optimization strategies. For specific use cases or performance issues, consult the troubleshooting section or create an issue in the repository.*
