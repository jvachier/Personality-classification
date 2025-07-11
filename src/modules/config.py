"""
Configuration constants and global parameters for the personality classification pipeline.
"""

import warnings
import logging
import sys

# Global parameters
RND = 42
N_SPLITS = 5

# Data augmentation parameters
ENABLE_DATA_AUGMENTATION = True  # Set to True to enable
AUGMENTATION_METHOD = (
    "sdv_copula"  # Options: "simple", "sdv_copula", "sdv_ctgan", "smotenc"
    # Note: Use "sdv_copula" on Apple Silicon - CTGAN may hang on M1/M2/M3
)
AUGMENTATION_RATIO = 0.05  # 5% additional synthetic data
QUALITY_THRESHOLD = 0.8  # For quality filtering
N_TRIALS_STACK = 15
N_TRIALS_BLEND = 200
LABEL_NOISE_RATE = 0.02

# Configure warnings
warnings.filterwarnings("ignore")
# Suppress joblib resource tracker warnings on macOS
warnings.filterwarnings("ignore", message="resource_tracker")

# Check for optional dependencies
try:
    import importlib.util
    SDV_AVAILABLE = importlib.util.find_spec("sdv") is not None
except ImportError:
    SDV_AVAILABLE = False

try:
    import importlib.util
    IMBLEARN_AVAILABLE = importlib.util.find_spec("imblearn") is not None
except ImportError:
    IMBLEARN_AVAILABLE = False


def setup_logging():
    """Configure logging for the personality classification pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("personality_classifier.log"),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # Log warnings about missing dependencies
    if not SDV_AVAILABLE:
        logger.warning("⚠️ SDV not available. Install with: pip install sdv")
    if not IMBLEARN_AVAILABLE:
        logger.warning(
            "⚠️ imbalanced-learn not available. Install with: pip install imbalanced-learn"
        )
    
    return logger
