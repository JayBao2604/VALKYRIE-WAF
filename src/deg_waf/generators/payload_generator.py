"""
Attack Payload Generator

This module generates attack payloads (SQLi, XSS, RCE, NoSQLi, SSRF) based on 
context-free grammar rules. It creates both original payloads and their tokenized 
representations for training security models.
"""

import random
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


class PayloadGenerator:
    """Generator for attack payloads using context-free grammar."""
    
    def __init__(self, grammar_file: str, attack_type: Optional[str] = None):
        """
        Initialize the payload generator.
        
        Args:
            grammar_file: Path to the grammar definition file
            attack_type: Type of attack (sqli, xss, rce, nosqli, ssrf) for labeling
        """
        self.attack_type = attack_type or Path(grammar_file).stem
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        self.grammar = self._load_grammar_from_file(grammar_file)
    
    def _load_grammar_from_file(self, file_name: str) -> Dict[str, List[str]]:
        """
        Load grammar rules from a file.
        
        Args:
            file_name: Path to the grammar file
            
        Returns:
            Dictionary mapping rule names to their expansions
        """
        grammar = {}
        current_rule = None

        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('//'):
                    continue
                line = line[:-1]
                
                # Check for a rule definition
                rule_match = re.match(r'([a-zA-Z0-9_]+)\s*:\s*(.*)', line)
                if rule_match:
                    current_rule = rule_match.group(1)
                    expansions = rule_match.group(2).split('|') 
                    
                    # Split different productions
                    grammar[current_rule] = [expansion.strip() for expansion in expansions]
                
                # Continuation of rules on the next line (sometimes multiline rules)
                elif current_rule:
                    expansions = line.split('|')
                    grammar[current_rule].extend([expansion.strip() for expansion in expansions])

        return grammar
    
    @staticmethod
    def _remove_first_last_quote(input_string: str) -> str:
        """Remove surrounding single quotes from a string."""
        if input_string.startswith("'") and input_string.endswith("'"):
            return input_string[1:-1]
        return input_string
    
    @staticmethod
    def _replace_escaped_quote(input_string: str) -> str:
        """Replace escaped quote with actual quote character."""
        if input_string == "\\'":
            return "'"
        return input_string
    
    def generate(self, rule: str = 'start') -> Tuple[str, List[int]]:
        """
        Generate a payload from a grammar rule.
        
        Args:
            rule: The grammar rule to expand
            
        Returns:
            Tuple of (original_payload, tokenized_payload)
        """
        if rule in self.grammar:
            expansion = random.choice(self.grammar[rule]).split()

            original_payload = []
            tokenized_payload = []

            for token in expansion:
                original, token_ids = self.generate(token)
                original_payload.append(original)
                tokenized_payload.extend(token_ids)

            return ''.join(original_payload), tokenized_payload
        else:
            processed_rule = self._replace_escaped_quote(self._remove_first_last_quote(rule))
            if processed_rule not in self.vocab:
                self.vocab[processed_rule] = len(self.vocab)

            return processed_rule, [self.vocab[processed_rule]]
    
    def token_ids_to_original_payload(self, tokenized_payload: List[int]) -> str:
        """
        Convert tokenized payload back to original string.
        
        Args:
            tokenized_payload: List of token IDs
            
        Returns:
            Original payload string
        """
        id_to_token = {v: k for k, v in self.vocab.items()}
        original_payload = [id_to_token[token_id] for token_id in tokenized_payload]
        return ''.join(original_payload)
    
    def verify_generation(self, num_tests: int = 100) -> bool:
        """
        Verify that generation and tokenization are consistent.
        
        Args:
            num_tests: Number of test generations to verify
            
        Returns:
            True if all tests pass, False otherwise
        """
        print(f"Verifying generation with {num_tests} tests...")
        for i in range(num_tests):
            sqli = self.generate('start')
            if sqli[0] != self.token_ids_to_original_payload(sqli[1]):
                print(f"❌ Verification failed at test {i}")
                return False
        print("✅ All verification tests passed")
        return True
    
    def generate_payloads(self, num_payloads: int = 10000, output_dir: str = '.') -> Tuple[List[str], List[List[int]]]:
        """
        Generate a dataset of unique attack payloads.
        
        Args:
            num_payloads: Number of unique payloads to generate
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (payloads, tokenized_payloads)
        """
        output = []
        tokenized_output = []
        payload_set: Set[str] = set()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_payloads} unique {self.attack_type.upper()} payloads...")
        
        for i in range(num_payloads):
            while True:
                payload = self.generate('start')
                if payload[0] not in payload_set:
                    payload_set.add(payload[0])
                    tokenized_output.append(payload[1])
                    output.append(f"{payload[0]}\n")
                    
                    # Verify consistency
                    if payload[0] != self.token_ids_to_original_payload(payload[1]):
                        print(f"⚠️ Inconsistency detected at payload {i}")
                    break
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_payloads} payloads...")
        
        # Create attack-type-specific directory
        attack_output_dir = f"{output_dir}/{self.attack_type}"
        Path(attack_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define output files
        output_file = f"{attack_output_dir}/{self.attack_type}.txt"
        tokenized_file = f"{attack_output_dir}/tokenized_{self.attack_type}.json"
        vocab_file = f"{attack_output_dir}/{self.attack_type}_vocab.json"
        
        # Write to files
        print(f"Saving payloads to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
        
        print(f"Saving tokenized payloads to {tokenized_file}...")
        with open(tokenized_file, 'w', encoding='utf-8') as f:
            json.dump(tokenized_output, f)
        
        print(f"Saving vocabulary to {vocab_file}...")
        with open(vocab_file, 'w', encoding='utf-8') as json_file:
            json.dump(self.vocab, json_file, indent=4)
        
        print(f"✅ Successfully generated {len(payload_set)} unique {self.attack_type.upper()} payloads")
        print(f"📊 Vocabulary size: {len(self.vocab)} tokens")
        
        return output, tokenized_output


def generate_all_attack_payloads(num_payloads: int = 15000, output_dir: str = 'generated_payloads'):
    """
    Generate payloads for all attack types.
    
    Args:
        num_payloads: Number of payloads to generate per attack type
        output_dir: Directory to save all generated payloads
    """
    # Define all attack types and their grammar files
    attack_types = {
        'sqli': 'data/grammars/sqli.txt',
        'xss': 'data/grammars/xss.txt',
        'cmdi': 'data/grammars/cmdi.txt',
        'nosqli': 'data/grammars/nosqli.txt',
        'ssrf': 'data/grammars/ssrf.txt'
    }
    
    print(f"{'='*60}")
    print(f"Generating {num_payloads} payloads for each attack type")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    for attack_type, grammar_file in attack_types.items():
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {attack_type.upper()}")
            print(f"{'='*60}")
            
            # Initialize generator for this attack type
            generator = PayloadGenerator(
                grammar_file=grammar_file,
                attack_type=attack_type
            )
            
            # Verify generation consistency
            generator.verify_generation(num_tests=100)
            
            # Generate payloads
            generator.generate_payloads(
                num_payloads=num_payloads,
                output_dir=output_dir
            )
            
            print(f"✅ {attack_type.upper()} generation completed!\n")
            
        except FileNotFoundError:
            print(f"⚠️ Grammar file not found: {grammar_file}")
            print(f"Skipping {attack_type.upper()}...\n")
        except Exception as e:
            print(f"❌ Error generating {attack_type.upper()} payloads: {e}\n")
    
    print(f"\n{'='*60}")
    print(f"🎉 All payload generation completed!")
    print(f"{'='*60}")


def main():
    """Main function to generate payloads for all attack types."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate attack payloads from grammar files')
    parser.add_argument('-n', '--num-payloads', type=int, default=15000,
                        help='Number of payloads to generate per attack type (default: 15000)')
    parser.add_argument('-o', '--output-dir', type=str, default='generated_payloads',
                        help='Output directory for generated payloads (default: generated_payloads)')
    parser.add_argument('-t', '--attack-type', type=str, choices=['sqli', 'xss', 'rce', 'nosqli', 'ssrf', 'all'],
                        default='all', help='Specific attack type to generate (default: all)')
    
    args = parser.parse_args()
    
    if args.attack_type == 'all':
        generate_all_attack_payloads(args.num_payloads, args.output_dir)
    else:
        # Generate for specific attack type
        grammar_file = f'grammars/{args.attack_type}.txt'
        generator = PayloadGenerator(grammar_file=grammar_file, attack_type=args.attack_type)
        generator.verify_generation(num_tests=100)
        generator.generate_payloads(num_payloads=args.num_payloads, output_dir=args.output_dir)


if __name__ == "__main__":
    main()