"""Reward model for evaluating generated attack payloads."""

import torch
import torch.nn as nn
from transformers import OPTConfig, OPTModel


class OPTRewardModel(nn.Module):
    """Reward model based on OPT architecture for evaluating generated payloads."""

    def __init__(self, config):
        super().__init__()
        self.opt = OPTModel(config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.opt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.linear(self.dropout(last_hidden)).squeeze(-1)
        return logits


def load_reward_model(path, device):
    """Load a pre-trained reward model from disk."""
    config = OPTConfig.from_pretrained(path)
    model = OPTRewardModel(config)
    model.load_state_dict(torch.load(f"{path}/pytorch_model.bin", map_location=device))
    model.eval()
    return model.to(device)


def get_reward_score(model, tokenizer, text, device):
    """Calculate reward score for a generated text using the reward model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        score = model(**inputs)
    return torch.sigmoid(score).item()
