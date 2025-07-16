# Six-Stack Personality Classification Pipeline

Production-ready machine learning pipeline for personality classification using ensemble learning, data augmentation, and automated hyperparameter optimization. Modular, maintainable, and includes an interactive dashboard.

## Technology Stack

**ML**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna  
**Data**: pandas, numpy, scipy, SDV  
**Dashboard**: Dash, Plotly  
**DevOps**: Docker, GitHub Actions, pre-commit, uv, Ruff, mypy, Bandit

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Ruff-orange.svg)](https://ruff.rs)
[![Dashboard](https://img.shields.io/badge/Dashboard-Dash-red.svg)](https://plotly.com/dash/)
[![Architecture](https://img.shields.io/badge/Architecture-Modular-purple.svg)](#-architecture)

## Dashboard Preview

![Dashboard Demo](https://github.com/user-attachments/assets/3541a036-0363-44cf-ae78-7be6b66b553e)

*Watch a live demo of the Personality Classification Dashboard in action*

## Quick Start

```bash
git clone <repository-url>
cd Personality-classification
uv sync
make train-models   # Train models
make dash           # Launch dashboard
uv run python src/main_modular.py   # Run pipeline
```

## Table of Contents

- [Dashboard Preview](#dashboard-preview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Model Stacks](#model-stacks)
- [Performance Metrics](#performance-metrics)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

## Features

- Modular architecture: 8 specialized modules
- 6 ensemble stacks (A-F) with complementary ML algorithms
- Automated hyperparameter optimization (Optuna)
- Advanced data augmentation (SDV Copula)
- Interactive Dash dashboard
- Dockerized deployment
- Full test coverage (pytest)

## Architecture

```
src/
├── main_modular.py                 # 🎯 Main production pipeline (MLOps-enhanced)
├── six_stack_personality_classifier.py  # 📚 Reference implementation
├── modules/                        # 🧩 Core modules
│   ├── config.py                   # ⚙️ Configuration & logging
│   ├── data_loader.py              # 📊 Data loading & external merge
│   ├── preprocessing.py            # 🔧 Feature engineering
│   ├── data_augmentation.py        # 🎲 Advanced synthetic data
│   ├── model_builders.py           # 🏭 Model stack construction
│   ├── ensemble.py                 # 🎯 Ensemble & OOF predictions
│   ├── optimization.py             # 🔍 Optuna utilities
│   └── utils.py                    # 🛠️ Utility functions

dash_app/                           # 🖥️ Interactive Dashboard
├── dashboard/                            # Application source
│   ├── app.py                      # Main Dash application
│   ├── layout.py                   # UI layout components
│   ├── callbacks.py                # Interactive callbacks
│   └── model_loader.py             # Model loading utilities
├── main.py                         # Application entry point
├── Dockerfile                      # Container configuration
└── docker-compose.yml             # Multi-service orchestration

models/                             # 🤖 Trained Models
├── ensemble_model.pkl              # Production ensemble model
├── ensemble_metadata.json         # Model metadata and labels
├── stack_*_model.pkl              # Individual stack models
└── stack_*_metadata.json          # Stack-specific metadata

scripts/                            # 🛠️ Utility Scripts
└── train_and_save_models.py        # Model training and persistence

data/                               # 📊 Datasets

docs/                               # 📝 Documentation
└── [Generated documentation]       # Technical guides

best_params/                        # 💾 Optimized parameters
└── stack_*_best_params.json        # Per-stack best parameters
```

## Installation

### Prerequisites

- **Python 3.11+**
- **uv** (modern Python package manager) - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Personality-classification

# Install dependencies
uv sync

# Verify installation
uv run python examples/minimal_test.py
```

### Alternative Installation (pip)

```bash
# If you prefer pip over uv
pip install -r requirements.txt  # Generated from pyproject.toml
```

## Usage

```bash
# Run production pipeline
uv run python src/main_modular.py

# Launch dashboard (after training models)
make train-models
make dash

# Stop dashboard
make stop-dash
```

## Dashboard

See the video demo above for the latest dashboard interface and features. To launch the dashboard:

```bash
make train-models
make dash
# Dashboard available at http://localhost:8050
```

## Configuration

The pipeline is highly configurable through `src/modules/config.py`:

### Core Parameters

```python
# Reproducibility
RND = 42                           # Global random seed

# Cross-validation
N_SPLITS = 5                       # Stratified K-fold splits

# Hyperparameter optimization
N_TRIALS_STACK = 15               # Optuna trials per stack (15 for testing, 100+ for production)
N_TRIALS_BLEND = 200              # Ensemble blending optimization trials

# Threading configuration
class ThreadConfig(Enum):
    N_JOBS = 4                    # Parallel jobs for sklearn
    THREAD_COUNT = 4              # Thread count for XGBoost/LightGBM
```

### Data Augmentation

```python
# Augmentation settings
ENABLE_DATA_AUGMENTATION = True
AUGMENTATION_METHOD = "sdv_copula"    # or "basic", "smote", "adasyn"
AUGMENTATION_RATIO = 0.05             # 5% synthetic data

# Quality control
DIVERSITY_THRESHOLD = 0.95            # Minimum diversity score
QUALITY_THRESHOLD = 0.7               # Minimum quality score
```

### Advanced Settings

```python
# Label noise for robustness
LABEL_NOISE_RATE = 0.02              # 2% label noise for Stack F

# Testing mode
TESTING_MODE = True                   # Reduced dataset for development
TESTING_SAMPLE_SIZE = 1000           # Samples in testing mode

# Logging
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR
```

## Model Stacks

The pipeline employs six specialized ensemble stacks, each optimized for different aspects of the problem:

| Stack | Focus                   | Algorithms                                                      | Hyperparameter Space         | Special Features            |
| ----- | ----------------------- | --------------------------------------------------------------- | ---------------------------- | --------------------------- |
| **A** | Traditional ML (Narrow) | Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost | Conservative search space    | Stable baseline performance |
| **B** | Traditional ML (Wide)   | Same as Stack A                                                 | Extended search space        | Broader exploration         |
| **C** | Gradient Boosting       | XGBoost, CatBoost                                               | Gradient boosting focused    | Tree-based specialists      |
| **D** | Sklearn Ensemble        | Extra Trees, Hist Gradient Boosting, SVM, Gaussian NB           | Sklearn-native models        | Diverse algorithm mix       |
| **E** | Neural Networks         | MLPClassifier, Deep architectures                               | Neural network tuning        | Non-linear pattern capture  |
| **F** | Noise-Robust Training   | Same as Stack A                                                 | Standard space + label noise | Improved generalization     |

### Ensemble Strategy

- **Out-of-fold predictions** for unbiased ensemble training
- **Optuna-optimized blending weights** for each stack
- **Meta-learning approach** with Logistic Regression as final combiner
- **Stratified cross-validation** ensures robust evaluation

## Performance Metrics

### Target Performance

The pipeline is designed to achieve high accuracy through ensemble learning and advanced optimization techniques. Performance will vary based on:

```
📊 Dataset Statistics
├── Training Samples: ~18,000+ (with augmentation)
├── Test Samples: ~6,000+
├── Original Features: 8 personality dimensions
├── Engineered Features: 14+ (with preprocessing)
├── Augmented Samples: Variable (adaptive, typically 5-10%)
└── Class Balance: Extrovert/Introvert classification

🔧 Technical Specifications
├── Memory Usage: <4GB peak (configurable)
├── CPU Utilization: 4 cores (configurable)
├── Model Persistence: ✅ Best parameters saved
└── Reproducibility: ✅ Fixed random seeds
```

## Testing & Validation

### Quick Validation

```bash
# Test installation and imports
uv run python examples/minimal_test.py

# Run lightweight demo
uv run python examples/main_demo.py

# Test individual modules
uv run python examples/test_modules.py
```

### Development Testing

```bash
# Enable testing mode (faster execution)
# Edit src/modules/config.py:
TESTING_MODE = True
TESTING_SAMPLE_SIZE = 1000

# Run with reduced dataset
uv run python src/main_modular.py
```

## Troubleshooting

### Common Issues

#### Dashboard Issues

```bash
# Dashboard won't start
make train-models              # Ensure models are trained first
make stop-dash && make dash    # Stop and restart dashboard

# Port already in use
lsof -ti:8050 | xargs kill     # Kill process on port 8050
make dash                      # Restart dashboard

# Missing model files
make train-models              # Retrain models
ls models/                     # Verify model files exist
```

#### Memory Issues

```bash
# Reduce computational load
# In src/modules/config.py:
N_TRIALS_STACK = 5          # Reduce from 15
ENABLE_DATA_AUGMENTATION = False
TESTING_MODE = True
```

#### Import Errors

```bash
# Verify environment
uv run python --version     # Should be 3.11+
uv sync                     # Reinstall dependencies
uv run python -c "import sklearn, pandas, numpy, dash; print('OK')"
```

#### Performance Issues

```bash
# Optimize for your system
# In src/modules/config.py:
class ThreadConfig(Enum):
    N_JOBS = 2              # Reduce from 4
    THREAD_COUNT = 2        # Reduce from 4
```

#### Optuna Crashes

```bash
# Use simpler models
uv run python examples/main_final.py  # Lightweight version
```

### Debug Mode

```bash
# Enable detailed logging
# In src/modules/config.py:
LOG_LEVEL = "DEBUG"

# Run with verbose output
uv run python src/main_modular.py 2>&1 | tee debug.log
```

## Documentation

See the `docs/` directory for:
- Technical Guide
- API Reference
- Data Augmentation
- Configuration Guide
- Performance Tuning
- Deployment Guide

## Lead Developer & Maintainer

**Lead Developer:** [Jeremy Vachier](https://github.com/jvachier)
For issues, feature requests, or questions, use GitHub Issues or Discussions.

## Contributing

Contributions welcome! Fork the repo, create a feature branch, implement and test your changes, then submit a pull request.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Project Status

**Status:** Production Ready | Interactive Dashboard | Modular | Well Documented
