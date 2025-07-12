# API Reference - Six-Stack Personality Classification Pipeline

## Module Overview

The pipeline consists of 8 core modules, each with well-defined interfaces and responsibilities.

## config.py

### Configuration Management

#### Global Constants
```python
RND: int = 42                    # Global random seed
N_SPLITS: int = 5               # Cross-validation folds
N_TRIALS_STACK: int = 15        # Optuna trials per stack
N_TRIALS_BLEND: int = 200       # Ensemble blending trials
LOG_LEVEL: str = "INFO"         # Logging level
```

#### Threading Configuration
```python
class ThreadConfig(Enum):
    """Centralized threading configuration."""
    N_JOBS: int = 4             # Parallel jobs for sklearn
    THREAD_COUNT: int = 4       # Thread count for XGB/LGB
```

#### Data Augmentation Configuration
```python
ENABLE_DATA_AUGMENTATION: bool = True
AUGMENTATION_METHOD: str = "sdv_copula"
AUGMENTATION_RATIO: float = 0.05
DIVERSITY_THRESHOLD: float = 0.95
QUALITY_THRESHOLD: float = 0.7
```

#### Functions
```python
def setup_logging() -> None:
    """Initialize structured logging configuration."""

def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance."""
```

## data_loader.py

### Data Loading and External Integration

#### Primary Functions
```python
def load_data_with_external_merge() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training/test data with external dataset merge using TOP-4 strategy.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            (train_df, test_df, submission_template)
    """

def merge_external_features(df_main: pd.DataFrame, 
                          df_external: pd.DataFrame,
                          is_training: bool = True) -> pd.DataFrame:
    """
    Merge external dataset features using strategic matching.
    
    Args:
        df_main: Primary dataset
        df_external: External dataset to merge
        is_training: Whether processing training data
        
    Returns:
        pd.DataFrame: Dataset with merged external features
    """
```

#### Utility Functions
```python
def validate_data_integrity(df: pd.DataFrame, 
                          data_type: str) -> dict[str, Any]:
    """
    Validate dataset integrity and return statistics.
    
    Args:
        df: Dataset to validate
        data_type: Type identifier ('train', 'test', 'external')
        
    Returns:
        dict[str, Any]: Validation statistics
    """
```

## preprocessing.py

### Data Preprocessing and Feature Engineering

#### Main Preprocessing Function
```python
def prep(df_tr: pd.DataFrame, 
         df_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Complete preprocessing pipeline with advanced competitive approach.
    
    Args:
        df_tr: Training dataframe
        df_te: Test dataframe
        
    Returns:
        tuple: (X_train, X_test, y_train, label_encoder)
    """
```

#### Feature Engineering Functions
```python
def correlation_based_imputation(df: pd.DataFrame, 
                               target_columns: list[str],
                               n_corr: int = 3) -> pd.DataFrame:
    """
    Impute missing values using correlation-based strategy.
    
    Args:
        df: Input dataframe
        target_columns: Columns to impute
        n_corr: Number of top correlated features to use
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """

def apply_one_hot_encoding(df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         categorical_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply one-hot encoding to categorical features.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe  
        categorical_features: List of categorical column names
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Encoded train and test dataframes
    """
```

## data_augmentation.py

### Advanced Data Augmentation

#### Main Augmentation Function
```python
def apply_data_augmentation(X: pd.DataFrame, 
                          y: pd.Series,
                          method: str = None,
                          ratio: float = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply adaptive data augmentation with quality control.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Augmentation method ('sdv_copula', 'smote', 'adasyn', 'basic')
        ratio: Augmentation ratio (0.0-1.0)
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: Augmented features and targets
    """
```

#### Augmentation Strategies
```python
def sdv_copula_augmentation(X: pd.DataFrame, 
                          y: pd.Series,
                          n_samples: int) -> tuple[pd.DataFrame, pd.Series]:
    """SDV Copula-based synthetic data generation."""

def smote_augmentation(X: pd.DataFrame,
                      y: pd.Series, 
                      n_samples: int) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE-based oversampling."""

def adasyn_augmentation(X: pd.DataFrame,
                       y: pd.Series,
                       n_samples: int) -> tuple[pd.DataFrame, pd.Series]:
    """ADASYN adaptive synthetic sampling."""
```

#### Quality Control
```python
def enhanced_quality_filtering(synthetic_data: pd.DataFrame,
                             original_data: pd.DataFrame,
                             quality_threshold: float = 0.7) -> tuple[pd.DataFrame, list[float]]:
    """
    Advanced quality filtering with multiple metrics.
    
    Args:
        synthetic_data: Generated synthetic samples
        original_data: Original training data
        quality_threshold: Minimum quality score
        
    Returns:
        tuple: (filtered_data, quality_scores)
    """

def calculate_diversity_score(data: pd.DataFrame) -> float:
    """Calculate dataset diversity score."""
```

## model_builders.py

### Model Stack Construction

#### Stack Builder Functions
```python
def build_stack(trial: optuna.Trial,
               seed: int = 42,
               wide_hp: bool = False) -> Pipeline:
    """
    Build traditional ML stack (A/B).
    
    Args:
        trial: Optuna trial for hyperparameter suggestion
        seed: Random seed
        wide_hp: Whether to use wide hyperparameter ranges
        
    Returns:
        Pipeline: Configured stacking classifier
    """

def build_stack_c(trial: optuna.Trial, seed: int = 42) -> Pipeline:
    """Build gradient boosting specialist stack (C)."""

def build_sklearn_stack(trial: optuna.Trial,
                       seed: int = 42,
                       X_full: pd.DataFrame = None) -> Pipeline:
    """Build sklearn ensemble stack (D)."""

def build_neural_stack(trial: optuna.Trial,
                      seed: int = 42,
                      X_full: pd.DataFrame = None) -> Pipeline:
    """Build neural network stack (E)."""

def build_noisy_stack(trial: optuna.Trial,
                     seed: int = 42,
                     noise_rate: float = 0.02) -> Pipeline:
    """Build noise-robust stack (F)."""
```

#### Model Configuration
```python
def get_base_models(trial: optuna.Trial,
                   seed: int,
                   wide_hp: bool = False) -> list[tuple[str, Any]]:
    """
    Get base models with optimized hyperparameters.
    
    Args:
        trial: Optuna trial
        seed: Random seed
        wide_hp: Use wide hyperparameter ranges
        
    Returns:
        list[tuple[str, Any]]: List of (name, model) tuples
    """
```

## ensemble.py

### Ensemble Learning and Out-of-Fold Predictions

#### OOF Generation
```python
def oof_probs(builder: callable,
             X: pd.DataFrame,
             y: pd.Series,
             X_test: pd.DataFrame,
             sample_weights: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold predictions for ensemble training.
    
    Args:
        builder: Model builder function
        X: Training features
        y: Training targets
        X_test: Test features
        sample_weights: Optional sample weights
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (oof_predictions, test_predictions)
    """

def oof_probs_noisy(builder: callable,
                   X: pd.DataFrame,
                   y: pd.Series,
                   X_test: pd.DataFrame,
                   noise_rate: float = 0.02,
                   sample_weights: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate OOF predictions with label noise injection."""
```

#### Blending Optimization
```python
def improved_blend_obj(trial: optuna.Trial,
                      oof_a: np.ndarray,
                      oof_b: np.ndarray,
                      oof_c: np.ndarray,
                      oof_d: np.ndarray,
                      oof_e: np.ndarray,
                      oof_f: np.ndarray,
                      y_true: pd.Series) -> float:
    """
    Objective function for ensemble weight optimization.
    
    Args:
        trial: Optuna trial
        oof_a through oof_f: Out-of-fold predictions from each stack
        y_true: True labels
        
    Returns:
        float: Accuracy score
    """
```

## optimization.py

### Hyperparameter Optimization and Utilities

#### Objective Function Creators
```python
def make_stack_objective(X: pd.DataFrame,
                        y: pd.Series,
                        seed: int = 42,
                        wide_hp: bool = False,
                        sample_weights: np.ndarray = None) -> callable:
    """Create objective function for traditional ML stacks."""

def make_stack_c_objective(X: pd.DataFrame,
                          y: pd.Series,
                          seed: int = 42,
                          sample_weights: np.ndarray = None) -> callable:
    """Create objective function for gradient boosting stack."""

def make_sklearn_stack_objective(X: pd.DataFrame,
                               y: pd.Series,
                               seed: int = 42,
                               sample_weights: np.ndarray = None) -> callable:
    """Create objective function for sklearn ensemble stack."""

def make_neural_stack_objective(X: pd.DataFrame,
                              y: pd.Series,
                              seed: int = 42,
                              sample_weights: np.ndarray = None) -> callable:
    """Create objective function for neural network stack."""

def make_noisy_stack_objective(X: pd.DataFrame,
                             y: pd.Series,
                             seed: int = 42,
                             noise_rate: float = 0.02,
                             sample_weights: np.ndarray = None) -> callable:
    """Create objective function for noise-robust stack."""
```

#### Parameter Management
```python
def save_best_trial_params(study: optuna.Study, filename: str) -> None:
    """
    Save best trial parameters to JSON file.
    
    Args:
        study: Completed Optuna study
        filename: Output filename (without extension)
    """

def load_best_trial_params(filename: str) -> dict | None:
    """
    Load best trial parameters from JSON file.
    
    Args:
        filename: Input filename (without extension)
        
    Returns:
        dict | None: Parameters dictionary or None if not found
    """
```

#### Utility Functions
```python
def add_label_noise(y: pd.Series,
                   noise_rate: float = 0.02,
                   random_state: int = 42) -> pd.Series:
    """
    Add controlled label noise for robustness training.
    
    Args:
        y: Original labels
        noise_rate: Fraction of labels to flip
        random_state: Random seed
        
    Returns:
        pd.Series: Labels with noise
    """
```

## utils.py

### Utility Functions and Helpers

#### Logging Utilities
```python
def setup_logging(level: str = "INFO") -> None:
    """Setup comprehensive logging configuration."""

def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance."""

def log_performance_metrics(metrics: dict[str, float],
                          prefix: str = "") -> None:
    """Log performance metrics in structured format."""
```

#### Data Utilities
```python
def calculate_data_characteristics(X: pd.DataFrame, 
                                 y: pd.Series) -> dict[str, Any]:
    """
    Calculate comprehensive data characteristics.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        dict[str, Any]: Data characteristics dictionary
    """

def validate_data_quality(X: pd.DataFrame,
                         y: pd.Series) -> dict[str, Any]:
    """Validate data quality and return issues."""
```

#### Model Utilities
```python
def calculate_model_complexity(model: Any) -> float:
    """Calculate model complexity score."""

def estimate_memory_usage(X: pd.DataFrame) -> str:
    """Estimate memory usage for dataset."""

def format_time_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
```

## Main Pipeline (main_modular.py)

### Core Pipeline Classes

#### Data Containers
```python
@dataclass
class StackConfig:
    """Configuration for a single stack in the ensemble."""
    name: str
    display_name: str
    seed: int
    objective_func: str
    sampler_startup_trials: int = 10
    wide_hp: bool | None = None
    noise_rate: float | None = None

class TrainingData(NamedTuple):
    """Container for training data."""
    X_full: pd.DataFrame
    X_test: pd.DataFrame
    y_full: pd.Series
    le: LabelEncoder
    submission: pd.DataFrame
```

#### Main Pipeline Functions
```python
def load_and_prepare_data(testing_mode: bool = True,
                         test_size: int = 1000) -> TrainingData:
    """Load and prepare all training data."""

def train_all_stacks(data: TrainingData) -> dict[str, optuna.Study]:
    """Train all 6 stacks with hyperparameter optimization."""

def create_model_builders(studies: dict[str, optuna.Study],
                         data: TrainingData) -> dict[str, callable]:
    """Create model builder functions for each stack."""

def generate_oof_predictions(builders: dict[str, callable],
                           data: TrainingData) -> dict[str, pd.Series]:
    """Generate out-of-fold predictions for all stacks."""

def optimize_ensemble_blending(oof_predictions: dict[str, pd.Series],
                             y_full: pd.Series) -> tuple[dict[str, float], float]:
    """Optimize ensemble blending weights."""

def refit_and_predict(builders: dict[str, callable],
                     best_weights: dict[str, float],
                     data: TrainingData) -> tuple[pd.DataFrame, str]:
    """Refit models on full data and generate final predictions."""

def main() -> None:
    """Main execution function for the pipeline."""
```

## Error Handling

### Exception Classes
```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""

class DataLoadingError(PipelineError):
    """Raised when data loading fails."""

class ModelTrainingError(PipelineError):
    """Raised when model training fails."""

class AugmentationError(PipelineError):
    """Raised when data augmentation fails."""
```

### Error Handling Patterns
```python
def with_retry(func: callable, 
               max_retries: int = 3,
               exceptions: tuple = (Exception,)) -> callable:
    """Decorator for automatic retry on failure."""

def handle_memory_error(func: callable) -> callable:
    """Decorator for graceful memory error handling."""
```

## Type Hints and Validation

### Common Types
```python
from typing import Union, Optional, Dict, List, Tuple, Any, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from optuna import Trial, Study

# Type aliases
ModelBuilder = Callable[[], BaseEstimator]
ObjectiveFunction = Callable[[Trial], float]
DataTuple = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Any]
```

### Input Validation
```python
def validate_dataframe(df: pd.DataFrame, 
                      required_columns: list[str] = None) -> None:
    """Validate dataframe structure and content."""

def validate_parameters(params: dict[str, Any],
                       schema: dict[str, type]) -> None:
    """Validate parameters against schema."""
```

---

*This API reference covers all public interfaces. For implementation details, see the source code and technical guide.*
