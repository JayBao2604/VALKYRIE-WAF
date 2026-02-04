"""
Simple script to download HuggingFace models with retry logic.
This helps when network connection is unstable.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def download_model(model_name: str, max_retries: int = 10):
    """Download model with retry logic."""
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Downloading {model_name}...")
            
            print("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ Tokenizer downloaded")
            
            print("Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print("✓ Model downloaded")
            
            print(f"\n✅ Successfully downloaded {model_name}!")
            print(f"   Model parameters: {model.num_parameters():,}")
            print(f"   Tokenizer vocab size: {len(tokenizer)}")
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Download interrupted. Progress saved. Try again...")
            if attempt < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            continue
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if attempt < max_retries:
                print(f"Retrying in 3 seconds...")
                time.sleep(3)
            continue
    
    print(f"\n❌ Failed to download after {max_retries} attempts")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace models")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Maximum retry attempts (default: 10)"
    )
    
    args = parser.parse_args()
    download_model(args.model_name, args.retries)
