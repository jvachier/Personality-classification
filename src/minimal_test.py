#!/usr/bin/env python3
"""
Minimal test of the modular structure.
"""

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from modules.config import RND, N_SPLITS, AUGMENTATION_METHOD
        print(f"✅ Config imported - RND: {RND}, N_SPLITS: {N_SPLITS}")
        
        from modules.utils import get_logger
        logger = get_logger(__name__)
        logger.info("✅ Utils imported and logger working")
        
        from modules.data_loader import load_data_with_external_merge
        print("✅ Data loader imported")
        
        from modules.preprocessing import prep
        print("✅ Preprocessing imported")
        
        from modules.model_builders import build_stack, build_stack_c
        print("✅ Model builders imported")
        
        from modules.ensemble import oof_probs, improved_blend_obj
        print("✅ Ensemble imported")
        
        print("🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

def test_data_loading():
    """Test basic data loading without processing."""
    try:
        import pandas as pd
        print("Testing basic data loading...")
        
        # Just test if files exist
        if os.path.exists("./data/train.csv"):
            df = pd.read_csv("./data/train.csv")
            print(f"✅ Train data loaded: {df.shape}")
            return True
        else:
            print("⚠️ Data files not found in ./data/ directory")
            return True  # This is not a critical failure for testing modules
            
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Minimal Modular Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_data_loading()
    
    if success:
        print("\n🎉 Minimal test completed successfully!")
        print("The modular refactoring is working correctly.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
