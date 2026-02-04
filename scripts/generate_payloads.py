"""
Script for generating attack payloads from grammar files.

This script uses context-free grammars to generate diverse attack payloads
for training and testing security models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deg_waf.generators import PayloadGenerator, generate_all_attack_payloads


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Generate attack payloads from grammar files')
    parser.add_argument('-n', '--num-payloads', type=int, default=15000,
                        help='Number of payloads to generate per attack type (default: 15000)')
    parser.add_argument('-o', '--output-dir', type=str, default='data/generated',
                        help='Output directory for generated payloads (default: data/generated)')
    parser.add_argument('-t', '--attack-type', type=str,
                        choices=['sqli', 'xss', 'cmdi', 'nosqli', 'ssrf', 'all'],
                        default='all',
                        help='Specific attack type to generate (default: all)')

    args = parser.parse_args()

    if args.attack_type == 'all':
        generate_all_attack_payloads(args.num_payloads, args.output_dir)
    else:
        # Generate for specific attack type
        grammar_file = f'data/grammars/{args.attack_type}.txt'
        generator = PayloadGenerator(grammar_file=grammar_file, attack_type=args.attack_type)
        generator.verify_generation(num_tests=100)
        generator.generate_payloads(num_payloads=args.num_payloads, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
