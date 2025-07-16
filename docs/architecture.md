# Architecture Documentation

## System Overview

The Six-Stack Personality Classification Pipeline is built with a modular, scalable architecture designed for machine learning competitions and production deployment.

### Core Architecture

- **Modular pipeline**: 8 core modules in `src/modules/`
- **Main pipeline**: `src/main_modular.py`
- **Dashboard**: `dash_app/` (Dash, Docker)
- **Model stacks**: 6 specialized ensembles (A-F)
- **Data flow**: Load → Preprocess → Augment → Train → Ensemble → Predict

## Model Stack Architecture

### Stack Specializations

| Stack | Type | Description |
|-------|------|-------------|
| **A** | Traditional ML (narrow) | Focused feature selection with classic algorithms |
| **B** | Traditional ML (wide) | Comprehensive feature engineering with traditional models |
| **C** | XGBoost/CatBoost | Gradient boosting specialists |
| **D** | Sklearn ensemble | Ensemble of sklearn algorithms |
| **E** | Neural networks | Deep learning approaches |
| **F** | Noise-robust | Robust methods for noisy data |

## Key Features

### Design Principles

- **Efficient**: Optimized for both speed and accuracy
- **Reproducible**: Consistent results with random seed control
- **Testable**: Comprehensive test coverage
- **Modular**: Easy to extend and maintain

### Core Capabilities

- **Full logging**: Comprehensive error handling and progress tracking
- **Data augmentation**: Advanced synthetic data generation
- **Hyperparameter optimization**: Automated tuning for each stack
- **Cross-validation**: Robust evaluation methodology
- **Ensemble learning**: Meta-learning for optimal predictions
