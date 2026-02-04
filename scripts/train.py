"""
Training script for A2C-based attack payload generation.

This script trains an A2C (Advantage Actor-Critic) model with GAE 
(Generalized Advantage Estimation) for generating attack payloads.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deg_waf.trainers import train_a2c
from deg_waf.utils import get_config


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Train A2C model for attack payload generation')
    parser.add_argument('-t', '--attack-type', type=str,
                        choices=['sqli', 'xss', 'cmdi', 'nosqli', 'ssrf'],
                        default='sqli',
                        help='Type of attack to train for (default: sqli)')
    parser.add_argument('-n', '--num-iters', type=int, default=500,
                        help='Number of training iterations (default: 500)')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--max-length', type=int, default=64,
                        help='Maximum sequence length (default: 64)')
    parser.add_argument('--policy-model', type=str,
                        help='Path to policy model (overrides default)')
    parser.add_argument('--reward-model', type=str,
                        help='Path to reward model (overrides default)')
    parser.add_argument('--vocab-path', type=str,
                        help='Path to vocabulary file (overrides default)')

    args = parser.parse_args()

    # Get config for attack type
    config = get_config(args.attack_type)

    # Override config with CLI arguments
    config['num_train_iters'] = args.num_iters
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['max_length'] = args.max_length

    if args.policy_model:
        config['policy_model_path'] = args.policy_model
    if args.reward_model:
        config['reward_model_path'] = args.reward_model
    if args.vocab_path:
        config['vocab_path'] = args.vocab_path

    print(f"{'='*60}")
    print(f"A2C Training Configuration for {args.attack_type.upper()}")
    print(f"{'='*60}")
    print(f"Attack Type: {config['attack_type']}")
    print(f"Iterations: {config['num_train_iters']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Max Length: {config['max_length']}")
    print(f"Policy Model: {config['policy_model_path']}")
    print(f"Reward Model: {config['reward_model_path']}")
    print(f"Vocab Path: {config['vocab_path']}")
    print(f"{'='*60}\n")

    # Start training
    train_a2c(config)


if __name__ == "__main__":
    main()
