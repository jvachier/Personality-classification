"""Configuration constants and global parameters for the personality classification pipeline."""

import logging
import sys
import warnings
from enum import Enum
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # Points to project root


class Paths(Enum):
    """File and directory paths for the project."""

    DATA_DIR = BASE_DIR / "data"
    SUBMISSIONS_DIR = BASE_DIR / "submissions"
    BEST_PARAMS_DIR = BASE_DIR / "best_params"
    CATBOOST_INFO_DIR = BASE_DIR / "catboost_info"
    LOGS_DIR = BASE_DIR / "logs"

    # Data files
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"
    SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
    PERSONALITY_DATASET_CSV = DATA_DIR / "personality_dataset.csv"

    # Log files
    PERSONALITY_CLASSIFIER_LOG = BASE_DIR / "personality_classifier.log"

    @property
    def value(self):
        """Return the path as a string."""
        return str(self._value_)

    def exists(self):
        """Check if the path exists."""
        return Path(self._value_).exists()

    def mkdir(self, parents=True, exist_ok=True):
        """Create the directory if it's a directory path."""
        Path(self._value_).mkdir(parents=parents, exist_ok=exist_ok)


class AugmentationMethod(Enum):
    """Enhanced data augmentation methods."""

    SIMPLE = "simple"
    SDV_COPULA = "sdv_copula"
    SDV_CTGAN = "sdv_ctgan"
    SMOTENC = "smotenc"
    # New advanced methods
    TVAE = "tvae"  # Variational Autoencoder
    FAST_ML = "fast_ml"  # Fast ML-based augmentation
    MIXED_ENSEMBLE = "mixed_ensemble"  # Combine multiple methods
    ADAPTIVE = "adaptive"  # Auto-select best method
    CLASS_BALANCED = "class_balanced"  # Balance classes intelligently


class TestingMode(Enum):
    """Testing mode configuration for development and debugging."""

    DISABLED = False
    ENABLED = True

    @property
    def value(self):
        """Return the actual boolean value."""
        return self._value_


class TestingConfig(Enum):
    """Testing configuration parameters."""

    TESTING_MODE = TestingMode.DISABLED  # Enable for development
    TESTING_SAMPLE_SIZE = 1000  # Number of samples to use in testing mode
    FULL_SAMPLE_SIZE = None  # Use full dataset when None

    @property
    def value(self):
        """Return the actual value."""
        return self._value_


class ModelConfig(Enum):
    """Model configuration parameters."""

    RND = 42
    N_SPLITS = 5
    N_TRIALS_STACK = 15  # Reduced for testing (original: 15)
    N_TRIALS_BLEND = 200  # Reduced for testing (original: 200)

    @property
    def value(self):
        """Return the actual value."""
        return self._value_


class ThreadConfig(Enum):
    """Thread and job configuration for parallel processing."""

    # Conservative values to prevent segmentation faults
    N_JOBS = 1  # Number of parallel jobs for scikit-learn models
    THREAD_COUNT = 1  # Number of threads for CatBoost models

    @property
    def value(self):
        """Return the actual value."""
        return self._value_


class AugmentationConfig(Enum):
    """Enhanced data augmentation configuration."""

    # Adaptive configuration
    ENABLE_DATA_AUGMENTATION = True
    AUGMENTATION_METHOD = AugmentationMethod.ADAPTIVE  # Auto-select best method

    # Dynamic ratios based on data characteristics
    MIN_AUGMENTATION_RATIO = 0.01
    BASE_AUGMENTATION_RATIO = 0.05  # 5% additional synthetic data
    MAX_AUGMENTATION_RATIO = 0.20

    # Adaptive scaling multipliers for user control
    ADAPTIVE_SCALING_ENABLED = True
    SIMPLE_AUGMENTATION_MULTIPLIER = 1.0  # Base multiplier for simple augmentation
    SDV_AUGMENTATION_MULTIPLIER = 1.2  # Slightly higher for SDV methods
    SMOTE_AUGMENTATION_MULTIPLIER = 0.8  # Lower for SMOTE-based methods
    ENSEMBLE_AUGMENTATION_MULTIPLIER = 1.5  # Higher for ensemble methods
    ADAPTIVE_QUALITY_MULTIPLIER = 1.3  # Quality-based scaling
    CLASS_BALANCE_MULTIPLIER = 1.1  # Class balancing multiplier

    # Quality and performance controls
    ENABLE_QUALITY_FILTERING = True
    QUALITY_THRESHOLD = 0.75  # For quality filtering
    ENABLE_DIVERSITY_CHECK = True
    DIVERSITY_THRESHOLD = 0.8

    # Performance optimizations
    MAX_AUGMENTATION_TIME_SECONDS = 300  # 5 minutes max
    ENABLE_CACHING = True
    CACHE_AUGMENTED_DATA = True

    # Class balancing
    ENABLE_CLASS_BALANCING = True
    TARGET_CLASS_BALANCE_RATIO = 0.7  # Aim for 70% balance minimum

    # Pseudo labelling configuration
    ENABLE_PSEUDO_LABELLING = True
    PSEUDO_CONFIDENCE_THRESHOLD = 0.95  # Minimum confidence for pseudo-labels
    PSEUDO_MAX_RATIO = 0.3  # Maximum ratio of pseudo-labels to original data

    # Legacy support
    AUGMENTATION_RATIO = 0.05  # Backward compatibility
    LABEL_NOISE_RATE = 0.02

    @property
    def value(self):
        """Return the actual value."""
        return self._value_


# Backward compatibility - expose values directly
RND = ModelConfig.RND.value
N_SPLITS = ModelConfig.N_SPLITS.value
N_TRIALS_STACK = ModelConfig.N_TRIALS_STACK.value
N_TRIALS_BLEND = ModelConfig.N_TRIALS_BLEND.value

# Thread configuration
N_JOBS = ThreadConfig.N_JOBS.value
THREAD_COUNT = ThreadConfig.THREAD_COUNT.value

# Enhanced data augmentation configuration exports
ENABLE_DATA_AUGMENTATION = AugmentationConfig.ENABLE_DATA_AUGMENTATION.value
AUGMENTATION_METHOD = AugmentationConfig.AUGMENTATION_METHOD.value.value
AUGMENTATION_RATIO = AugmentationConfig.AUGMENTATION_RATIO.value
BASE_AUGMENTATION_RATIO = AugmentationConfig.BASE_AUGMENTATION_RATIO.value
MIN_AUGMENTATION_RATIO = AugmentationConfig.MIN_AUGMENTATION_RATIO.value
MAX_AUGMENTATION_RATIO = AugmentationConfig.MAX_AUGMENTATION_RATIO.value
QUALITY_THRESHOLD = AugmentationConfig.QUALITY_THRESHOLD.value
ENABLE_QUALITY_FILTERING = AugmentationConfig.ENABLE_QUALITY_FILTERING.value
ENABLE_DIVERSITY_CHECK = AugmentationConfig.ENABLE_DIVERSITY_CHECK.value
DIVERSITY_THRESHOLD = AugmentationConfig.DIVERSITY_THRESHOLD.value
ENABLE_CLASS_BALANCING = AugmentationConfig.ENABLE_CLASS_BALANCING.value
TARGET_CLASS_BALANCE_RATIO = AugmentationConfig.TARGET_CLASS_BALANCE_RATIO.value
MAX_AUGMENTATION_TIME_SECONDS = AugmentationConfig.MAX_AUGMENTATION_TIME_SECONDS.value
ENABLE_CACHING = AugmentationConfig.ENABLE_CACHING.value
CACHE_AUGMENTED_DATA = AugmentationConfig.CACHE_AUGMENTED_DATA.value
LABEL_NOISE_RATE = AugmentationConfig.LABEL_NOISE_RATE.value

# Adaptive scaling multipliers
ADAPTIVE_SCALING_ENABLED = AugmentationConfig.ADAPTIVE_SCALING_ENABLED.value
SIMPLE_AUGMENTATION_MULTIPLIER = AugmentationConfig.SIMPLE_AUGMENTATION_MULTIPLIER.value
SDV_AUGMENTATION_MULTIPLIER = AugmentationConfig.SDV_AUGMENTATION_MULTIPLIER.value
SMOTE_AUGMENTATION_MULTIPLIER = AugmentationConfig.SMOTE_AUGMENTATION_MULTIPLIER.value
ENSEMBLE_AUGMENTATION_MULTIPLIER = (
    AugmentationConfig.ENSEMBLE_AUGMENTATION_MULTIPLIER.value
)
ADAPTIVE_QUALITY_MULTIPLIER = AugmentationConfig.ADAPTIVE_QUALITY_MULTIPLIER.value
CLASS_BALANCE_MULTIPLIER = AugmentationConfig.CLASS_BALANCE_MULTIPLIER.value

# Pseudo labelling configuration
ENABLE_PSEUDO_LABELLING = AugmentationConfig.ENABLE_PSEUDO_LABELLING.value
PSEUDO_CONFIDENCE_THRESHOLD = AugmentationConfig.PSEUDO_CONFIDENCE_THRESHOLD.value
PSEUDO_MAX_RATIO = AugmentationConfig.PSEUDO_MAX_RATIO.value

# Testing configuration
TESTING_MODE = TestingConfig.TESTING_MODE.value.value
TESTING_SAMPLE_SIZE = TestingConfig.TESTING_SAMPLE_SIZE.value
FULL_SAMPLE_SIZE = TestingConfig.FULL_SAMPLE_SIZE.value

# Data augmentation library availability checks
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


# Initialize directories
def initialize_directories():
    """Create necessary directories if they don't exist."""
    for path_enum in [
        Paths.DATA_DIR,
        Paths.SUBMISSIONS_DIR,
        Paths.BEST_PARAMS_DIR,
        Paths.CATBOOST_INFO_DIR,
        Paths.LOGS_DIR,
    ]:
        path_enum.mkdir()


# Configure warnings and logging
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
    # Ensure logs directory exists
    Paths.LOGS_DIR.mkdir()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Paths.PERSONALITY_CLASSIFIER_LOG.value),
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


# Initialize on import
initialize_directories()
