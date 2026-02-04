# tools/reward_model_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_reward_model(model_path: str, device="cpu"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def score_payload(prompt: str, response_ids, tokenizer, model, device="cpu", max_length=128):
    text = prompt + tokenizer.decode(response_ids[0], skip_special_tokens=True)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, num_labels)

        # === SAFE HANDLING ===
        if logits.ndim == 2 and logits.shape[0] == 1:
            reward = torch.sigmoid(logits[0][0]).item()
        elif logits.ndim == 1:
            reward = torch.sigmoid(logits[0]).item()
        else:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    return reward