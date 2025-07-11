"""
Utility functions for the personality classification pipeline.
"""

import numpy as np
import logging


def add_label_noise(y, noise_rate=0.02, random_state=42):
    """
    Add controlled label noise for regularization.
    
    Args:
        y: Target labels
        noise_rate: Fraction of labels to flip (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        y_noisy: Labels with added noise
    """
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_indices = np.random.choice(len(y), n_flip, replace=False)

    # Flip labels (0->1, 1->0)
    y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]

    return y_noisy


def get_logger(name=__name__):
    """Get a logger instance."""
    return logging.getLogger(name)
