# DEG-WAF - Kaggle Setup Guide

Simple guide to run pretraining on Kaggle with OPT-125M.

## Quick Start on Kaggle

### 1. Upload Project to Kaggle

1. Create a new Kaggle notebook
2. Upload the `data/generated` folder to Kaggle Datasets
3. Upload the `scripts/pretrain.py` file

### 2. Install Dependencies

```python
!pip install transformers datasets torch accelerate
```

### 3. Run Training

Basic command:
```bash
!python pretrain.py --attack-type sqli
```

With custom settings:
```bash
!python pretrain.py \
    --attack-type sqli \
    --model-name facebook/opt-125m \
    --num-epochs 3 \
    --batch-size 8 \
    --learning-rate 5e-5
```

For quick testing:
```bash
!python pretrain.py --attack-type sqli --max-samples 1000 --num-epochs 1
```

### 4. Training Arguments

**Required:**
- `--attack-type`: Choose from `sqli`, `xss`, `cmdi`, `nosqli`, `ssrf`

**Optional:**
- `--model-name`: Default is `facebook/opt-125m`
- `--num-epochs`: Default is 3
- `--batch-size`: Default is 8 (adjust based on GPU memory)
- `--learning-rate`: Default is 5e-5
- `--max-samples`: Limit training samples (for testing)
- `--fp16`: Enable mixed precision (faster but may have issues)

### 5. Expected Results

**Small test (1000 samples, 1 epoch):**
- Time: ~25-30 seconds
- Final eval loss: ~3.6

**Full training (all samples, 3 epochs):**
- SQLi: ~46k samples, ~30-40 minutes
- XSS: ~39k samples, ~25-35 minutes
- Others: Similar times based on dataset size

### 6. Output

Trained model saved to:
```
models/pretrained/{attack_type}/opt-125m/
```

Files created:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer_config.json` - Tokenizer settings
- `metadata.json` - Training metadata

### 7. Using Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/pretrained/sqli/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("models/pretrained/sqli/opt-125m")

# Generate payloads
prompt = "SELECT * FROM"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, num_return_sequences=5)

for i, output in enumerate(outputs):
    print(f"{i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")
```

## Kaggle GPU Settings

- **T4 GPU (16GB)**: Can use batch size 16-32 with FP16
- **P100 GPU (16GB)**: Can use batch size 16-32 with FP16
- **CPU only**: Use batch size 4, will be slower

## Common Issues

### Out of Memory (OOM)
Reduce batch size:
```bash
--batch-size 4
```

### Slow Training
Enable FP16 (if supported):
```bash
--fp16
```

### Missing Data Files
Make sure you uploaded the correct folder structure:
```
data/
  generated/
    sqli/
      sqli.txt
      sqli_vocab.json
    xss/
      xss.txt
      xss_vocab.json
    ...
```

## Attack Types Available

1. **sqli** - SQL Injection (46,005 samples)
2. **xss** - Cross-Site Scripting (39,088 samples)
3. **cmdi** - Command Injection
4. **nosqli** - NoSQL Injection
5. **ssrf** - Server-Side Request Forgery

## Next Steps

After pretraining:
1. Use model for RL-based adversarial training
2. Fine-tune with A2C/PPO against WAF
3. Generate novel bypass payloads
4. Evaluate against real WAFs

## Notes

- Default uses FP32 (more stable on Kaggle)
- Use `--fp16` flag for faster training if your GPU supports it
- Training is simple and straightforward - no complex optimizations needed
- Works out of the box on Kaggle environment
