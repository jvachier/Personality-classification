"""
Configuration constants and global parameters for the personality classification pipeline.
"""

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
    """Data augmentation methods."""

    SIMPLE = "simple"
    SDV_COPULA = "sdv_copula"
    SDV_CTGAN = "sdv_ctgan"
    SMOTENC = "smotenc"


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

    TESTING_MODE = TestingMode.ENABLED  # Enable for development
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
    """Data augmentation configuration."""

    ENABLE_DATA_AUGMENTATION = True
    AUGMENTATION_METHOD = AugmentationMethod.SDV_COPULA  # Default method
    AUGMENTATION_RATIO = 0.05  # 5% additional synthetic data
    QUALITY_THRESHOLD = 0.8  # For quality filtering
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

ENABLE_DATA_AUGMENTATION = AugmentationConfig.ENABLE_DATA_AUGMENTATION.value
AUGMENTATION_METHOD = AugmentationConfig.AUGMENTATION_METHOD.value.value
AUGMENTATION_RATIO = AugmentationConfig.AUGMENTATION_RATIO.value
QUALITY_THRESHOLD = AugmentationConfig.QUALITY_THRESHOLD.value
LABEL_NOISE_RATE = AugmentationConfig.LABEL_NOISE_RATE.value

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
