#!/usr/bin/env python3
"""
Test script to verify the modular structure works correctly.
"""


def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")

    try:
        from modules.config import setup_logging, RND, N_SPLITS

        print("✅ config module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")

    try:
        from modules.utils import get_logger, add_label_noise

        print("✅ utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")

    try:
        from modules.data_loader import load_data_with_external_merge

        print("✅ data_loader module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_loader: {e}")

    try:
        from modules.data_augmentation import apply_data_augmentation

        print("✅ data_augmentation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_augmentation: {e}")

    try:
        from modules.preprocessing import prep

        print("✅ preprocessing module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import preprocessing: {e}")

    try:
        from modules.model_builders import build_stack

        print("✅ model_builders module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import model_builders: {e}")

    try:
        from modules.optimization import save_best_trial_params

        print("✅ optimization module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import optimization: {e}")

    try:
        from modules.ensemble import oof_probs

        print("✅ ensemble module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ensemble: {e}")


def test_logging():
    """Test that logging setup works."""
    print("\nTesting logging setup...")
    try:
        from modules.config import setup_logging

        logger = setup_logging()
        logger.info("Logging test successful!")
        print("✅ Logging setup works correctly")
    except Exception as e:
        print(f"❌ Logging setup failed: {e}")


def test_configuration():
    """Test that configuration values are accessible."""
    print("\nTesting configuration...")
    try:
        from modules.config import RND, N_SPLITS, AUGMENTATION_METHOD

        print(f"✅ RND = {RND}")
        print(f"✅ N_SPLITS = {N_SPLITS}")
        print(f"✅ AUGMENTATION_METHOD = {AUGMENTATION_METHOD}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


if __name__ == "__main__":
    print("🧪 Testing Modular Personality Classification Pipeline")
    print("=" * 55)

    test_imports()
    test_logging()
    test_configuration()

    print("\n🎉 Module testing completed!")
    print("\nNext steps:")
    print("1. Install required dependencies (pandas, sklearn, xgboost, etc.)")
    print("2. Place your data files in the data/ directory")
    print("3. Run: python main_modular.py")
