"""Unit tests for payload generator."""

import pytest
from deg_waf.generators import PayloadGenerator


def test_payload_generator_init():
    """Test PayloadGenerator initialization."""
    generator = PayloadGenerator(grammar_file='data/grammars/sqli.txt', attack_type='sqli')
    assert generator.attack_type == 'sqli'
    assert len(generator.vocab) >= 4  # At least special tokens


def test_generate_payload():
    """Test payload generation."""
    generator = PayloadGenerator(grammar_file='data/grammars/xss.txt', attack_type='xss')
    payload, tokens = generator.generate('start')
    
    assert isinstance(payload, str)
    assert isinstance(tokens, list)
    assert len(payload) > 0
    assert len(tokens) > 0


def test_token_conversion():
    """Test token to payload conversion."""
    generator = PayloadGenerator(grammar_file='data/grammars/sqli.txt', attack_type='sqli')
    original, tokens = generator.generate('start')
    reconstructed = generator.token_ids_to_original_payload(tokens)
    
    assert original == reconstructed
