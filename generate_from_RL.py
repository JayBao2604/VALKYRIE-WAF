import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm

# === CONFIG ===
model_path = "models/finetune-models/opt-a2c-SQLI"
num_payloads = 1000
output_file = "outputs/generated_payloads.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD TOKENIZER + MODEL ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

# === PROMPT SOURCE (hoặc dùng prompt mặc định) ===
# Bạn có thể thay bằng danh sách prompt từ file, hoặc một prompt cố định
prompts = ["SQLi attack: "] * num_payloads

# === GENERATION FUNCTION ===
def generate_payload(prompt, max_length=200, temperature=0.8, top_p=0.95):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === GENERATE ===
with open(output_file, 'w', encoding='utf-8') as f:
    for prompt in tqdm(prompts, desc="Generating payloads"):
        text = generate_payload(prompt)
        payload = text.replace("SQLi attack:", "").strip()
        f.write(payload + "\n")

print(f"\n✅ Generated {num_payloads} payloads saved to {output_file}")