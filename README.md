# Personality Classification Pipeline

A production-ready, modular machine learning pipeline for personality classification with ensemble learning and Optuna optimization.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run the main production pipeline (all 6 stacks)
uv run python src/main_modular.py

# Or run working examples
uv run python examples/main_final.py    # Lightweight version (0.97+ accuracy)
uv run python examples/main_demo.py     # Demo version with dummy models
uv run python examples/minimal_test.py  # Verify installation
```

## 📁 Project Structure

```
├── src/
│   ├── main_modular.py                 # 🎯 Main production pipeline
│   ├── six_stack_personality_classifier.py  # Original monolithic file
│   └── modules/                        # 🏗️ Modular components
│       ├── config.py                   # Configuration & logging
│       ├── data_loader.py              # Data loading & external merge
│       ├── preprocessing.py            # Data preprocessing
│       ├── data_augmentation.py        # Data augmentation methods
│       ├── model_builders.py           # Model stack builders
│       ├── ensemble.py                 # Ensemble & OOF predictions
│       ├── optimization.py             # Optuna parameter utilities
│       └── utils.py                    # Utility functions
├── examples/                           # 📚 Example scripts
│   ├── main_final.py                   # Lightweight working example
│   ├── main_demo.py                    # Demo with dummy models
│   ├── test_modules.py                 # Module testing
│   └── minimal_test.py                 # Import verification
├── data/                               # 📊 Dataset files
└── docs/                               # 📝 Documentation
```

## 🎯 Features

### ✅ **Complete Modular Architecture**
- **8 specialized modules** with single responsibility
- **Clean separation of concerns** for maintainability
- **Independent testing** of each component

### ✅ **Advanced ML Pipeline**
- **6 specialized stacks** (A-F) with different algorithms
- **Optuna hyperparameter optimization** for each stack
- **Ensemble blending** with optimized weights
- **Pseudo-labeling** for additional training data
- **Cross-validation** with stratified folds

### ✅ **Production Ready**
- **Professional logging** throughout the pipeline
- **Error handling** and timeout protection
- **Parameter persistence** for reproducibility
- **Configurable settings** via config module

### ✅ **High Performance**
- **97%+ accuracy** on personality classification
- **Multiple model types**: Random Forest, Logistic Regression, XGBoost, Neural Networks
- **Data augmentation** with SDV synthetic data generation
- **External data integration** using TOP-4 solution strategy

## 🔧 Configuration

Edit `src/modules/config.py` to customize:

```python
# Global parameters
RND = 42                    # Random seed
N_SPLITS = 5               # Cross-validation folds
N_TRIALS_STACK = 15        # Optuna trials per stack
N_TRIALS_BLEND = 200       # Optuna trials for blending

# Data augmentation
ENABLE_DATA_AUGMENTATION = True
AUGMENTATION_METHOD = "sdv_copula"
AUGMENTATION_RATIO = 0.05

# Model training
LABEL_NOISE_RATE = 0.02
```

## 📊 Pipeline Overview

1. **Data Loading** - Load training/test data with external dataset merge
2. **Preprocessing** - Feature engineering, imputation, encoding
3. **Data Augmentation** - Generate synthetic samples (optional)
4. **Stack Training** - Train 6 specialized model stacks with Optuna
5. **Ensemble Optimization** - Optimize blending weights
6. **Pseudo-labeling** - Add high-confidence predictions (optional)
7. **Final Predictions** - Generate ensemble predictions

## 🎯 Model Stacks

| Stack | Description | Models |
|-------|-------------|---------|
| **A** | Traditional ML (narrow) | Random Forest, Logistic Regression |
| **B** | Traditional ML (wide) | Random Forest with broader search |
| **C** | Gradient Boosting | XGBoost, CatBoost |
| **D** | Sklearn Ensemble | Extra Trees, AdaBoost, SVM |
| **E** | Neural Networks | MLPClassifier, Deep Learning |
| **F** | Noisy Training | Models trained with label noise |

## 📈 Results

Latest performance metrics:
- **Ensemble Accuracy**: 97.01%
- **Individual Stack Range**: 96.86% - 96.98%
- **Training Samples**: 18,524
- **Test Samples**: 6,175
- **Features**: 14 (after preprocessing)

## 🧪 Testing

```bash
# Test module imports
uv run python examples/test_modules.py

# Minimal verification
uv run python examples/minimal_test.py

# Lightweight demo
uv run python examples/main_demo.py
```

## 🔍 Troubleshooting

### Memory Issues
- Reduce `N_TRIALS_STACK` in config
- Disable data augmentation: `ENABLE_DATA_AUGMENTATION = False`
- Use lightweight examples in `examples/` folder

### Import Errors
- Ensure all dependencies installed: `uv sync`
- Check Python environment: `uv run python --version`

### Optuna Crashes
- Use simpler models in `examples/main_final.py`
- Reduce trial counts in configuration

## 📚 Documentation

- [`MODULAR_SUCCESS.md`](MODULAR_SUCCESS.md) - Complete refactoring summary
- [`REFACTORING_COMPLETE.md`](REFACTORING_COMPLETE.md) - Technical details
- [`src/modules/README.md`](src/modules/README.md) - Module documentation

## 🤝 Contributing

1. Work on individual modules in `src/modules/`
2. Add tests in `examples/`
3. Update configuration in `src/modules/config.py`
4. Test with `uv run python examples/test_modules.py`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 Success Metrics

✅ **Modular Architecture**: 8 focused modules  
✅ **High Performance**: 97%+ accuracy  
✅ **Production Ready**: Professional logging & error handling  
✅ **Maintainable**: Clean separation of concerns  
✅ **Scalable**: Easy to add new models/features  
✅ **Documented**: Comprehensive documentation  

---

**Status**: ✅ **Production Ready** | **Performance**: 🚀 **97%+ Accuracy** | **Architecture**: 🏗️ **Fully Modular**