"""
Reward Model Loader
===================
Loads trained OPT reward models and scores payloads
Based on OPT_reward_model.ipynb architecture
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTModel, OPTConfig
from pathlib import Path
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OPTRewardModel(nn.Module):
    """OPT-based Reward Model - Same architecture as training"""
    
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.opt = OPTModel(config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.opt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # Take last token
        logits = self.linear(self.dropout(last_hidden)).squeeze(-1)
        
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        
        return logits


class RewardModelScorer:
    """
    Reward Model Scorer
    Loads trained reward models and scores payloads
    """
    
    def __init__(self, 
                 attack_type: str,
                 model_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize reward model scorer
        
        Args:
            attack_type: sqli, xss, rce, ssrf, xxe, nosqli
            model_dir: Path to reward model directory
            device: cpu or cuda
        """
        self.attack_type = attack_type
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None
        self.model_path = None
        
        # Find model directory if not specified
        if model_dir is None:
            model_dir = self._find_model_path(attack_type)
        
        if model_dir and Path(model_dir).exists():
            self.model_path = model_dir
            self._load_model(model_dir)
        else:
            logger.warning(f"Reward model not found for {attack_type}")
    
    def _find_model_path(self, attack_type: str) -> Optional[str]:
        """
        Find reward model path by checking common locations
        Priority: modsecurity/sampler > safeline/sampler > others
        """
        base_paths = [
            Path("models/reward-models/modsecurity/sampler"),
            Path("models/reward-models/safeline/sampler"),
            Path("models/reward-models/modsecurity/no_sampler"),
            Path("models/reward-models/safeline/no_sampler"),
        ]
        
        for base_path in base_paths:
            model_path = base_path / f"opt125m-reward_{attack_type}"
            if model_path.exists():
                logger.info(f"Found reward model: {model_path}")
                return str(model_path)
        
        # Also check for gpt2 models
        for base_path in base_paths:
            model_path = base_path / f"gpt2-reward_{attack_type}"
            if model_path.exists():
                logger.info(f"Found reward model: {model_path}")
                return str(model_path)
        
        return None
    
    def _load_model(self, model_path: str):
        """Load reward model from path"""
        try:
            model_path = Path(model_path)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✓ Loaded tokenizer from {model_path}")
            
            # Load config
            config = OPTConfig.from_pretrained(str(model_path))
            config.pad_token_id = self.tokenizer.eos_token_id
            
            # Create model
            self.model = OPTRewardModel(config)
            
            # Load weights
            pytorch_model_bin = model_path / "pytorch_model.bin"
            if pytorch_model_bin.exists():
                state_dict = torch.load(str(pytorch_model_bin), map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✓ Loaded model weights from {pytorch_model_bin}")
            else:
                logger.warning(f"pytorch_model.bin not found in {model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model ready on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            self.model = None
            self.tokenizer = None
    
    def score_payload(self, payload: str, max_length: int = 128) -> float:
        """
        Score a single payload
        
        Args:
            payload: The payload string to score
            max_length: Maximum token length (default 128 from training)
        
        Returns:
            Score between 0 and 1 (sigmoid of logits)
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Model not loaded, returning 0.5")
            return 0.5
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                payload,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(**inputs)
            
            # Convert to probability (sigmoid of logits)
            score = torch.sigmoid(logits).cpu().item()
            
            return score
        
        except Exception as e:
            logger.error(f"Error scoring payload: {e}")
            return 0.5
    
    def score_batch(self, payloads: list, max_length: int = 128) -> list:
        """
        Score multiple payloads
        
        Args:
            payloads: List of payload strings
            max_length: Maximum token length
        
        Returns:
            List of scores
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Model not loaded, returning default scores")
            return [0.5] * len(payloads)
        
        scores = []
        try:
            # Tokenize all payloads
            inputs = self.tokenizer(
                payloads,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(**inputs)
            
            # Convert to probabilities
            scores = torch.sigmoid(logits).cpu().squeeze().tolist()
            
            # Handle single payload case
            if not isinstance(scores, list):
                scores = [scores]
            
            return scores
        
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            return [0.5] * len(payloads)
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None and self.tokenizer is not None


def test_reward_scorer():
    """Test reward model scorer with sample payloads"""
    
    print("\n" + "="*80)
    print("TESTING REWARD MODEL SCORER")
    print("="*80)
    
    # Test different attack types
    attack_types = ["sqli", "xss", "rce"]
    test_payloads = {
        "sqli": [
            "1' OR '1'='1",
            "1' UNION SELECT * FROM users --",
            "admin' OR 1=1 --",
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror='alert(1)'>",
            "<svg onload=alert('xss')>",
        ],
        "rce": [
            "; ls -la",
            "| whoami",
            "; cat /etc/passwd",
        ]
    }
    
    for attack_type in attack_types:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING {attack_type.upper()} REWARD MODEL")
        print(f"{'='*80}")
        
        # Initialize scorer
        scorer = RewardModelScorer(
            attack_type=attack_type,
            device="cpu"
        )
        
        if not scorer.is_loaded():
            print(f"❌ Reward model not found for {attack_type}")
            continue
        
        print(f"✓ Model loaded: {scorer.model_path}")
        
        # Get test payloads
        payloads = test_payloads.get(attack_type, [])
        
        print(f"\n📥 Scoring {len(payloads)} {attack_type.upper()} payloads:\n")
        print("-" * 80)
        
        # Score each payload
        for i, payload in enumerate(payloads, 1):
            score = scorer.score_payload(payload)
            quality = "🌟 EXCELLENT" if score >= 0.85 else \
                      "✅ GOOD" if score >= 0.70 else \
                      "⚠️ ACCEPTABLE" if score >= 0.50 else \
                      "❌ POOR"
            
            print(f"\n[{i}] {quality}")
            print(f"    Payload: {payload[:70]}")
            print(f"    Score: {score:.4f}")
        
        # Batch score
        print(f"\n{'-'*80}")
        print("📊 Batch Scoring:")
        scores = scorer.score_batch(payloads)
        for i, score in enumerate(scores, 1):
            print(f"  [{i}] {score:.4f}")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n  Average Score: {avg_score:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_reward_scorer()
