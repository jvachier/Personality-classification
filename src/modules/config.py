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
N_TRIALS_STACK = 3  # Reduced for testing (original: 15)
N_TRIALS_BLEND = 10  # Reduced for testing (original: 200)
LABEL_NOISE_RATE = 0.02

# Data augmentation imports
try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTENC

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Configure warnings
warnings.filterwarnings("ignore")
# Suppress joblib resource tracker warnings on macOS
warnings.filterwarnings("ignore", message="resource_tracker")

# Suppress verbose SDV logging
logging.getLogger("sdv").setLevel(logging.WARNING)
logging.getLogger("rdt").setLevel(logging.WARNING)
logging.getLogger("copulas").setLevel(logging.WARNING)
logging.getLogger("sdv.single_table.base").setLevel(logging.WARNING)
logging.getLogger("SYNTHESIZER").setLevel(logging.WARNING)
logging.getLogger("SingleTableSynthesizer").setLevel(logging.WARNING)
logging.getLogger("MultiTableSynthesizer").setLevel(logging.WARNING)
logging.getLogger("sdv.data_processing").setLevel(logging.WARNING)
logging.getLogger("sdv.metadata").setLevel(logging.WARNING)


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
