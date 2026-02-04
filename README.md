# DEG-WAF: Deep Learning Web Application Firewall

A reinforcement learning-based framework for generating and detecting attack payloads using Advantage Actor-Critic (A2C) algorithms with Generalized Advantage Estimation (GAE).

## Features

- **Attack Payload Generation**: Grammar-based generation for SQLi, XSS, RCE, NoSQLi, and SSRF attacks
- **Reinforcement Learning**: A2C with GAE for training language models to generate effective payloads
- **Modular Architecture**: Clean separation of models, trainers, generators, and utilities
- **CLI Tools**: Easy-to-use scripts for training and payload generation
- **RAG Integration**: Retrieval Augmented Generation for vulnerability analysis

## Project Structure

```
deg-waf/
├── src/deg_waf/          # Main package
│   ├── models/           # Neural network models (reward, value)
│   ├── trainers/         # Training algorithms (A2C agent, trainer)
│   ├── generators/       # Payload generators
│   ├── utils/            # Utilities (GAE, config)
│   └── rag/              # RAG modules (crawler, analyzer)
├── data/
│   ├── grammars/         # Grammar files for attack types
│   └── generated/        # Generated payloads and vocabularies
├── models/
│   ├── pretrained/       # Pre-trained language models
│   ├── finetuned/        # Fine-tuned models
│   └── rewards/          # Reward models
├── scripts/              # CLI scripts
│   ├── train.py          # Training script
│   └── generate_payloads.py  # Payload generation script
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
└── configs/              # Configuration files
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DEG-WAF

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Generate Attack Payloads

```bash
# Generate all attack types
python scripts/generate_payloads.py

# Generate specific attack type
python scripts/generate_payloads.py -t xss -n 10000

# Custom output directory
python scripts/generate_payloads.py -o custom/output/dir
```

### Train A2C Model

```bash
# Train for SQL injection
python scripts/train.py -t sqli -n 1000

# Train for XSS with custom parameters
python scripts/train.py -t xss --num-iters 2000 --batch-size 8

# Train with custom models
python scripts/train.py -t rce --policy-model path/to/model --reward-model path/to/reward
```

## Attack Types Supported

- **SQLi** - SQL Injection
- **XSS** - Cross-Site Scripting
- **RCE** - Remote Code Execution
- **NoSQLi** - NoSQL Injection
- **SSRF** - Server-Side Request Forgery

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- pandas
- tqdm

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]

## Citation

If you use this work, please cite:

```bibtex
@software{deg_waf,
  title={DEG-WAF: Deep Learning Web Application Firewall},
  author={DEG-WAF Team},
  year={2025}
}
```
