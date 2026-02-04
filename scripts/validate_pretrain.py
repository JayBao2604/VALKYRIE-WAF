"""Quick validation script for pretrain.py optimizations."""

import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if file exists."""
    if Path(path).exists():
        size = Path(path).stat().st_size
        print(f"✓ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def main():
    print("\n" + "="*60)
    print("PRETRAIN VALIDATION")
    print("="*60 + "\n")
    
    all_good = True
    
    # Check data files
    print("📁 Data Files:")
    all_good &= check_file_exists("data/generated/sqli/sqli.txt", "SQLi payloads")
    all_good &= check_file_exists("data/generated/sqli/sqli_vocab.json", "SQLi vocab")
    
    print("\n📜 Scripts:")
    all_good &= check_file_exists("scripts/pretrain.py", "Pretrain script")
    
    print("\n📚 Documentation:")
    all_good &= check_file_exists("docs/PRETRAIN_OPTIMIZATIONS.md", "Optimizations guide")
    
    print("\n🔧 Test Import:")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        all_good = False
    
    try:
        from transformers import AutoTokenizer
        print(f"✓ Transformers: OK")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        all_good = False
    
    try:
        from datasets import Dataset
        print(f"✓ Datasets: OK")
    except Exception as e:
        print(f"✗ Datasets import failed: {e}")
        all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to train!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix above issues")
    print("="*60 + "\n")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
