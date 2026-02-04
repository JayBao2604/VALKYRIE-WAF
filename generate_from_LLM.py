import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os

# ========== CẤU HÌNH ==========
model_path = "models/pretrain-models/gpt2_xss"
output_file = "data/stage1/gpt2_xss_payloads.txt"
num_samples = 10000
max_new_tokens = 64

# Tự động chọn device (ưu tiên MPS → CUDA → CPU)
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Tắt cảnh báo tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== LOAD MODEL ==========
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ========== KHỞI TẠO INPUT ==========
start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
input_ids = torch.tensor([[start_token_id]]).to(device)
attention_mask = torch.ones_like(input_ids)

# ========== SINH PAYLOAD ==========
generated_payloads = []

with torch.no_grad():
    for _ in tqdm(range(num_samples), desc="Generating payloads"):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        # Giải mã và xoá khoảng trắng
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        payload = gen_text.replace(" ", "").strip()
        generated_payloads.append(payload)

# ========== GHI FILE ==========
with open(output_file, "w", encoding="utf-8") as f:
    for payload in generated_payloads:
        f.write(payload + "\n")

print(f"[+] Đã sinh {len(generated_payloads)} payload XSS (dính liền) và lưu vào: {output_file}")
