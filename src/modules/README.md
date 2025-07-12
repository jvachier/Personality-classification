# Personality Classification Pipeline - Modular Version

This directory contains the refactored, modular version of the six-stack personality classification pipeline.

## Project Structure

```
src/
├── modules/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Global constants and configuration
│   ├── utils.py                 # Utility functions (logging, noise addition)
│   ├── data_loader.py           # Data loading and external merge functions
│   ├── data_augmentation.py     # Data augmentation methods (SDV, SMOTE, etc.)
│   ├── preprocessing.py         # Data preprocessing and pseudo-labeling
│   ├── model_builders.py        # Stack building functions for different models
│   ├── optimization.py          # Parameter optimization utilities
│   └── ensemble.py              # Out-of-fold predictions and blending
├── main_modular.py              # Main execution script (modular version)
└── six_stack_personality_classifier.py  # Original monolithic script
```

## Module Overview

### `config.py`

- Global parameters and constants (RND, N_SPLITS, etc.)
- Data augmentation configuration
- Dependency checking (SDV, imbalanced-learn)
- Logging setup function

### `data_loader.py`

- `load_data_with_external_merge()`: Loads and merges external personality data using advanced merge strategy

### `data_augmentation.py`

- `simple_mixed_augmentation()`: Basic noise-based augmentation
- `sdv_augmentation()`: High-quality synthetic data using SDV (GaussianCopula, CTGAN)
- `smotenc_augmentation()`: SMOTE for mixed numerical/categorical data
- `apply_data_augmentation()`: Main augmentation dispatcher function

### `preprocessing.py`

- `prep()`: Main preprocessing function with advanced competitive approach
- `add_pseudo_labeling_conservative()`: Conservative pseudo-labeling
- `create_domain_balanced_dataset()`: Domain weighting for distribution alignment

### `model_builders.py`

- `build_stack()`: Main XGBoost + LightGBM + CatBoost stack
- `build_stack_c()`: XGBoost + CatBoost combination
- `build_sklearn_stack()`: RandomForest + ExtraTrees + HistGradientBoosting
- `build_neural_stack()`: MLPClassifier + SVM + NaiveBayes ensemble
- `build_noisy_stack()`: Stack trained on noisy labels for regularization

### `optimization.py`

- `save_best_trial_params()`: Save Optuna trial parameters to JSON
- `load_best_trial_params()`: Load saved parameters for warm starts

### `ensemble.py`

- `oof_probs()`: Generate out-of-fold predictions using cross-validation
- `oof_probs_noisy()`: OOF predictions with noisy labels
- `improved_blend_obj()`: Optuna objective for optimizing ensemble weights

### `utils.py`

- `add_label_noise()`: Add controlled label noise for regularization
- `get_logger()`: Get logger instance

## Usage

Run the modular version:

```bash
cd src
python main_modular.py
```

## Key Features

1. **Modular Architecture**: Clean separation of concerns across multiple modules
2. **Maintainability**: Each module has a specific responsibility
3. **Reusability**: Functions can be imported and used independently
4. **Configuration Management**: Centralized configuration in `config.py`
5. **Error Handling**: Proper exception handling throughout
6. **Logging**: Comprehensive logging using Python's logging module
7. **Documentation**: Extensive docstrings and type hints

## Benefits of Modular Approach

1. **Code Organization**: Easier to navigate and understand
2. **Testing**: Individual modules can be unit tested
3. **Debugging**: Easier to isolate and fix issues
4. **Collaboration**: Team members can work on different modules
5. **Extensibility**: Easy to add new models or augmentation methods
6. **Version Control**: Better git history and conflict resolution

## Dependencies

The same dependencies as the original script:

- pandas, numpy, sklearn
- xgboost, lightgbm, catboost
- optuna
- Optional: sdv, imbalanced-learn

## Performance

The modular version maintains the same functionality and performance as the original monolithic script while providing better code organization and maintainability.
