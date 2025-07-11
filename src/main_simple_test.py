#!/usr/bin/env python3
"""
Simple test script for the modular personality classification pipeline.
Tests core functionality without heavy ML operations.
"""

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import RND, N_SPLITS, AUGMENTATION_METHOD
from modules.data_loader import load_data_with_external_merge
from modules.preprocessing import prep
from modules.utils import add_label_noise, get_logger

logger = get_logger(__name__)


def test_data_pipeline():
    """Test the data loading and preprocessing pipeline."""
    logger.info("🧪 Testing Data Pipeline")
    logger.info("=" * 50)
    
    try:
        # Test data loading
        logger.info("📊 Testing data loading...")
        train_df, test_df, submission = load_data_with_external_merge()
        logger.info(f"✅ Data loaded - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission.shape}")
        
        # Test preprocessing
        logger.info("🔧 Testing preprocessing...")
        X_train, X_test, y_train, label_encoder = prep(train_df, test_df)
        logger.info(f"✅ Preprocessing completed - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # Test label noise (small amount)
        logger.info("🔄 Testing label noise addition...")
        y_noisy = add_label_noise(y_train.copy(), noise_rate=0.01)
        changed_labels = (y_train != y_noisy).sum()
        logger.info(f"✅ Label noise added - {changed_labels} labels changed")
        
        logger.info("🎉 Data pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {str(e)}")
        return False


def test_configurations():
    """Test configuration values."""
    logger.info("🧪 Testing Configurations")
    logger.info("=" * 50)
    
    logger.info(f"Random seed: {RND}")
    logger.info(f"CV splits: {N_SPLITS}")
    logger.info(f"Augmentation method: {AUGMENTATION_METHOD}")
    
    logger.info("✅ Configuration test completed!")


if __name__ == "__main__":
    logger.info("🎯 Simple Modular Pipeline Test")
    logger.info("=" * 50)
    
    # Test configurations
    test_configurations()
    
    # Test data pipeline
    success = test_data_pipeline()
    
    if success:
        logger.info("🎉 All tests passed! Modular refactoring is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
