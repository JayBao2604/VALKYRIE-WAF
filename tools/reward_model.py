import torch
from transformers import AutoTokenizer, OPTConfig, OPTModel
import os
import torch.nn as nn

class OPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.opt = OPTModel(config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.opt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # last token
        logits = self.linear(self.dropout(last_hidden)).squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits

# === CONFIG ===
MODEL_DIR = "models/reward-models/opt125m-reward_sqli" 
MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.bin")
MODEL_NAME = "facebook/opt-125m"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# === Load config ===
config = OPTConfig.from_pretrained(MODEL_NAME)
config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# === Load model ===
model = OPTRewardModel(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === Scoring function ===
def score_payload(payload: str) -> float:
    """
    Return a continuous reward score (probability of class=1) in [0.0, 1.0].
    """
    inputs = tokenizer(
        payload,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs)
        if isinstance(logits, tuple):  
            logits = logits[1]
        score = torch.sigmoid(logits).item()
    return score