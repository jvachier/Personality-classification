# Architecture Documentation

## Overview

This project implements a modular personality classification pipeline with ensemble learning and hyperparameter optimization.

## Component Architecture

### Core Modules (`src/modules/`)

1. **config.py** - Configuration management and logging setup
2. **data_loader.py** - Data loading with external dataset merging
3. **preprocessing.py** - Feature engineering and data preprocessing
4. **data_augmentation.py** - Data augmentation strategies
5. **model_builders.py** - Model construction for different stacks
6. **ensemble.py** - Out-of-fold predictions and ensemble methods
7. **optimization.py** - Optuna hyperparameter optimization utilities
8. **utils.py** - General utility functions

### Execution Scripts

- **src/main_modular.py** - Main production pipeline
- **examples/main_final.py** - Lightweight working example
- **examples/main_demo.py** - Demo with simplified models
- **examples/test_modules.py** - Module testing script
- **examples/minimal_test.py** - Import verification

## Data Flow

1. **Data Loading** → External dataset merge → Feature extraction
2. **Preprocessing** → Feature engineering → Data augmentation
3. **Model Training** → 6 specialized stacks with Optuna optimization
4. **Ensemble** → Out-of-fold predictions → Blend optimization
5. **Pseudo-labeling** → Conservative high-confidence labeling
6. **Final Prediction** → Weighted ensemble → Submission generation

## Stack Configurations

- **Stack A**: Traditional ML (narrow hyperparameters)
- **Stack B**: Traditional ML (wide hyperparameters)
- **Stack C**: XGBoost + CatBoost specialized
- **Stack D**: Sklearn ensemble models
- **Stack E**: Neural network models
- **Stack F**: Noisy label training

## Performance Features

- Memory-efficient processing
- CPU-optimized configurations
- Robust error handling with timeouts
- Modular testing capabilities
- Comprehensive logging
