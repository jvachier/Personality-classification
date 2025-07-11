"""
Optimization functions and parameter utilities for the personality classification pipeline.
"""

import os
import json
from .utils import get_logger

logger = get_logger(__name__)


def save_best_trial_params(study, model_name, params_dir="best_params"):
    """Save the best trial parameters to a JSON file."""
    os.makedirs(params_dir, exist_ok=True)
    best_params = study.best_trial.params
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    with open(filepath, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters for {model_name} to {filepath}")
    return best_params


def load_best_trial_params(model_name, params_dir="best_params"):
    """Load the best trial parameters from a JSON file."""
    filepath = os.path.join(params_dir, f"{model_name}_best_params.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            params = json.load(f)
        logger.info(f"Loaded best parameters for {model_name} from {filepath}")
        return params
    else:
        logger.info(f"No saved parameters found for {model_name} at {filepath}")
        return None
