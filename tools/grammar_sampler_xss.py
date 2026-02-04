import random
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tools.reward_model import score_payload

RuleMap = Dict[str, List[List[str]]]
WeightsMap = Dict[Tuple[str, Tuple[str, ...]], float]

def load_grammar(grammar_file: str) -> str:
    with open(grammar_file, 'r', encoding='utf-8') as f:
        return f.read()

def load_vocab(vocab_file: str) -> Set[str]:
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return set(json.load(f).keys())

def build_rule_map(grammar_text: str) -> RuleMap:
    rule_map: RuleMap = {}
    for line in grammar_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if ':' in line:
            head, body = line.split(':', 1)
            head = head.strip()
            body = body.split(';')[0]
            exps = [alt.strip() for alt in body.split('|')]
            rule_map[head] = [exp.split() for exp in exps]
    return rule_map

def init_weights(rule_map: RuleMap, init_weight: float = 1.0) -> WeightsMap:
    return {
        (head, tuple(exp)): init_weight
        for head, exps in rule_map.items() for exp in exps
    }

def update_weights(used: List[Tuple[str, Tuple[str, ...]]], weights: WeightsMap, reward: float, lr: float = 0.1):
    for key in used:
        if key in weights:
            weights[key] *= (1.0 + lr * (reward - 0.5))

def track_expansions(symbol: str, rule_map: RuleMap, weights: WeightsMap,
                     used: List[Tuple[str, Tuple[str, ...]]]) -> str:
    if symbol not in rule_map:
        return symbol
    exps = rule_map[symbol]
    ws = [weights.get((symbol, tuple(exp)), 1.0) for exp in exps]
    idx = random.choices(range(len(exps)), weights=ws, k=1)[0]
    chosen = exps[idx]
    used.append((symbol, tuple(chosen)))
    return ''.join(track_expansions(tok, rule_map, weights, used) for tok in chosen)

def generate_from_previous(symbol: str, rule_map: RuleMap, weights: WeightsMap,
                           prev_used: Set[Tuple[str, Tuple[str, ...]]],
                           used: List[Tuple[str, Tuple[str, ...]]],
                           retain_prob: float = 0.7) -> str:
    if symbol not in rule_map:
        return symbol
    exps = rule_map[symbol]
    candidates = [exp for exp in exps if (symbol, tuple(exp)) in prev_used]
    if candidates and random.random() < retain_prob:
        chosen = random.choice(candidates)
    else:
        ws = [weights.get((symbol, tuple(exp)), 1.0) for exp in exps]
        chosen = random.choices(exps, weights=ws, k=1)[0]
    used.append((symbol, tuple(chosen)))
    return ''.join(generate_from_previous(tok, rule_map, weights, prev_used, used, retain_prob)
                   for tok in chosen)

def tokenize_payload(payload: str, vocab: Set[str]) -> List[str]:
    tokens = []
    i = 0
    while i < len(payload):
        match = None
        for j in range(len(payload), i, -1):
            sub = payload[i:j]
            if sub in vocab:
                match = sub
                break
        if match:
            tokens.append(match)
            i += len(match)
        else:
            tokens.append(payload[i])
            i += 1
    return tokens

def normalize_payload(payload: str) -> str:
    return re.sub(r'\s+', '', payload)

def detect_attack_root_symbol(payload: str) -> str:
    lower = payload.lower()
    if "onerror" in lower or "onload" in lower or "onclick" in lower:
        return "eventContext"
    if "javascript" in lower or "alert" in lower or "<script" in lower:
        return "attriContext"
    return "start"

def refine_if_needed(original_payload: str,
                     original_score: float,
                     grammar_file: str = None,
                     vocab_file: str = None,
                     score_threshold: float = 0.8,
                     max_attempts: int = 15,
                     retain_prob: float = 0.7) -> Tuple[str, float]:
    if original_score >= score_threshold:
        return original_payload, original_score

    base_dir = Path(__file__).parent.resolve()
    grammar_file = grammar_file or (base_dir / "xss" / "xss-grammar.txt")
    vocab_file = vocab_file or (base_dir / "xss" / "xss_vocab.json")

    rule_map = build_rule_map(load_grammar(grammar_file))
    vocab = load_vocab(vocab_file)
    weights = init_weights(rule_map)

    best_payload = original_payload
    best_score = original_score
    best_used: List[Tuple[str, Tuple[str, ...]]] = []
    root_symbol = detect_attack_root_symbol(original_payload)

    for _ in range(max_attempts):
        used: List[Tuple[str, Tuple[str, ...]]] = []
        if best_used:
            payload = generate_from_previous(root_symbol, rule_map, weights, set(best_used), used, retain_prob)
        else:
            payload = track_expansions(root_symbol, rule_map, weights, used)

        if any(tok not in vocab for tok in tokenize_payload(payload, vocab)):
            continue

        score = score_payload(payload)
        if score > best_score:
            best_payload, best_score = payload, score
            best_used = used.copy()
        update_weights(used, weights, score)

        if best_score >= score_threshold:
            break

    return normalize_payload(best_payload), best_score
