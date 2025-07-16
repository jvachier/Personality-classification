# Architecture Documentation

## Architecture Overview

- Modular pipeline: 8 core modules in `src/modules/`
- Main pipeline: `src/main_modular.py`
- Dashboard: `dash_app/` (Dash, Docker)
- Model stacks: 6 specialized ensembles (A-F)
- Data flow: Load → Preprocess → Augment → Train → Ensemble → Predict

## Stacks
- A: Traditional ML (narrow)
- B: Traditional ML (wide)
- C: XGBoost/CatBoost
- D: Sklearn ensemble
- E: Neural networks
- F: Noise-robust

## Key Features
- Efficient, reproducible, and testable
- Full logging and error handling
