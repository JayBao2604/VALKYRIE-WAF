# VALKYRIE-WAF 🛡️

> **Advanced Web Application Firewall Bypass Framework using Deep Reinforcement Learning and Large Language Models**

## 🎥 Demo Video

<video src="https://github.com/JayBao2604/VALKYRIE-WAF/raw/main/%5BDemo%5D-VALKYRIE-WAF.mkv" controls width="100%">
  Your browser does not support the video tag.
</video>

*Complete demonstration of VALKYRIE-WAF in action - showing payload generation, WAF bypass testing, and validation pipeline*

---

VALKYRIE-WAF is a cutting-edge research framework that leverages reinforcement learning (A2C/PPO) and transformer-based language models to generate sophisticated web attack payloads capable of bypassing modern WAF systems. The framework combines grammar-based generation, reward modeling, and advanced post-processing validation to create production-ready offensive security testing tools.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Key Features

### 🤖 **Reinforcement Learning-Based Generation**
- **A2C (Advantage Actor-Critic)** with Generalized Advantage Estimation (GAE) for optimal payload generation
- **PPO (Proximal Policy Optimization)** for stable policy updates
- Custom reward models trained on real-world WAF bypass patterns
- Value network for accurate advantage estimation

### 🎓 **Large Language Model Integration**
- Fine-tuned **Facebook OPT-125M** and **FLAN-T5** models for attack payload generation
- Transformer-based architecture for context-aware payload synthesis
- Support for multiple attack vectors through specialized model training

### ✅ **Advanced Payload Validation System**
- **Grammar-based validation** against formal attack syntax rules
- **Semantic analysis** to ensure functional validity
- **WAF evasion analysis** detecting 15+ obfuscation techniques
- **Multi-dimensional scoring** (grammar, reward, evasion, semantic)
- **Automatic error correction** while preserving bypass techniques

### 🎯 **Comprehensive Attack Coverage**
- **SQL Injection (SQLi)** - Boolean-based, UNION-based, Time-based, Error-based
- **Cross-Site Scripting (XSS)** - Reflected, Stored, DOM-based
- **Remote Code Execution (RCE)** - Command injection, Code evaluation
- **NoSQL Injection (NoSQLi)** - MongoDB, CouchDB attack patterns
- **Server-Side Request Forgery (SSRF)** - Internal service exploitation
- **XML External Entity (XXE)** - XML parser attacks

### 🔍 **WAF Testing & Evaluation**
- **ModSecurity** bypass testing
- **SafeLine** WAF evasion
- Cross-WAF validation with extensive result notebooks
- Real-world deployment simulation

### 📊 **RAG-Enhanced Vulnerability Analysis**
- Automated **vulnerability disclosure crawler** (Bugcrowd, HackerOne)
- **Retrieval-Augmented Generation** for context-aware attack synthesis
- Historical vulnerability pattern analysis

## 📁 Project Structure

```
VALKYRIE-WAF/
├── src/deg_waf/              # Core framework package
│   ├── models/               # Neural network architectures
│   │   ├── reward_model.py   # OPT-based reward scoring
│   │   └── value_network.py  # Value head for advantage estimation
│   ├── trainers/             # RL training algorithms
│   │   ├── a2c_agent.py      # A2C agent implementation
│   │   └── a2c_trainer.py    # Training loop with GAE
│   ├── generators/           # Payload generation engines
│   │   └── payload_generator.py  # Grammar-based generation
│   ├── utils/                # Utility functions
│   │   ├── gae.py            # Generalized Advantage Estimation
│   │   └── config.py         # Configuration management
│   └── rag/                  # RAG components
│       ├── crawler.py        # Vulnerability disclosure crawler
│       └── analyzer.py       # Semantic payload analysis
│
├── data/
│   ├── grammars/             # Attack grammar definitions
│   │   ├── sqli.txt          # SQL Injection patterns
│   │   ├── xss.txt           # XSS attack patterns
│   │   ├── cmdi.txt          # Command injection patterns
│   │   ├── nosqli.txt        # NoSQL injection patterns
│   │   └── ssrf.txt          # SSRF patterns
│   └── generated/            # Generated payloads and vocabularies
│
├── models/                   # Trained model checkpoints
│   ├── pretrained/           # Base language models
│   ├── finetuned/            # Fine-tuned attack models
│   └── rewards/              # Reward model checkpoints
│
├── scripts/                  # Command-line tools
│   ├── train.py              # RL training script
│   ├── generate_payloads.py  # Payload generation
│   ├── pretrain.py           # Model pretraining
│   └── validate_pretrain.py  # Validation script
│
├── notebooks/                # Experimental notebooks
│   ├── results_rl_*.ipynb    # WAF bypass results (ModSecurity, SafeLine)
│   ├── rl-a2c-*.ipynb        # A2C training experiments
│   └── rl-ppo-*.ipynb        # PPO training experiments
│
├── tests/                    # Unit tests
│   └── test_generators.py
│
├── tools/                    # Additional utilities
│   ├── grammar_sampler_*.py  # Grammar-based sampling tools
│   ├── reward_model.py       # Reward calculation
│   └── waf_tester.py         # WAF testing utilities
│
├── advanced_post_rl_agent.py    # Advanced payload validator
├── demo_post_rl_validation.py   # Validation demo
├── generate_from_RL.py          # RL-based generation
├── generate_from_LLM.py         # LLM-based generation
├── reward_model_loader.py       # Reward model utilities
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JayBao2604/VALKYRIE-WAF.git
cd VALKYRIE-WAF

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

#### 1. Generate Attack Payloads

```bash
# Generate SQL injection payloads
python scripts/generate_payloads.py -t sqli -n 10000

# Generate XSS payloads
python scripts/generate_payloads.py -t xss -n 5000

# Generate all attack types
python scripts/generate_payloads.py
```

#### 2. Train RL Models

```bash
# Train A2C model for SQL injection
python scripts/train.py -t sqli --num-iters 1000 --batch-size 8

# Train with custom reward model
python scripts/train.py -t xss \
    --policy-model facebook/opt-125m \
    --reward-model models/rewards/xss-reward \
    --learning-rate 5e-5
```

#### 3. Pretrain Language Models

```bash
# Pretrain on SQL injection dataset
python scripts/pretrain.py --attack-type sqli --num-epochs 3

# Quick test with limited samples
python scripts/pretrain.py --attack-type xss --max-samples 1000 --num-epochs 1
```

#### 4. Advanced Payload Validation

```python
from advanced_post_rl_agent import AdvancedPostRLAgent

# Initialize validator
agent = AdvancedPostRLAgent(
    attack_type="sqli",
    min_combined_score=0.70,
    device="cpu"
)

# Validate payload
result = agent.validate_payload("1' OR '1'='1")

if result.is_production_ready:
    print(f"✓ Production ready: {result.corrected_payload}")
    print(f"Quality Score: {result.quality_score.combined_score:.4f}")
else:
    print(f"✗ Rejected: {result.rejection_reasons}")

# Batch processing
payloads = ["1' OR '1'='1", "admin' --", "1' UNION SELECT * --"]
results = agent.process_batch(payloads, verbose=True)
agent.print_summary()
```

## 🧪 Training Pipeline

### Stage 1: Pretraining
1. Load base language model (OPT-125M / FLAN-T5)
2. Fine-tune on grammar-generated attack payloads
3. Add attack-specific vocabulary tokens
4. Save pretrained checkpoint

### Stage 2: Reward Model Training
1. Collect successful/failed WAF bypass examples
2. Train binary classifier on OPT architecture
3. Optimize for bypass detection accuracy
4. Save reward model checkpoint

### Stage 3: RL Fine-tuning (A2C/PPO)
1. Initialize policy from pretrained model
2. Generate payload trajectories
3. Calculate rewards using reward model
4. Compute advantages using GAE
5. Update policy and value networks
6. Iterate until convergence

### Stage 4: Post-Processing Validation
1. Grammar syntax validation
2. Semantic analysis (attack method detection)
3. WAF evasion technique identification
4. Error correction with obfuscation preservation
5. Multi-dimensional quality scoring
6. Production readiness classification

## 📊 Evaluation & Results

The framework includes comprehensive evaluation notebooks in the `notebooks/` directory:

- **WAF Bypass Results**: `results_rl_*_cross_modsec.ipynb`, `results_rl_*_cross_safeline.ipynb`
- **Attack-Specific Results**: `results_rl_sqli.ipynb`, `results_rl_xss.ipynb`, `results_rl_rce.ipynb`
- **Training Experiments**: `rl-a2c-*.ipynb`, `rl-ppo-*.ipynb`

## 🛠️ Advanced Features

### Grammar-Based Generation
- Formal grammar definitions for each attack type
- Context-free grammar (CFG) sampling
- Token-based payload construction
- Special vocabulary management

### Reward Modeling
- OPT-based binary classification
- Sigmoid activation for 0-1 reward range
- Trained on real-world WAF bypass data
- Support for custom reward functions

### GAE (Generalized Advantage Estimation)
- Lambda parameter for bias-variance tradeoff
- Temporal difference learning
- Normalized advantage values
- Improved policy gradient estimates

### Post-RL Validation
- **Grammar Score** (35%): Syntax correctness
- **Reward Score** (35%): Model evaluation
- **Evasion Score** (15%): WAF bypass capability
- **Semantic Score** (15%): Attack feasibility
- Combined scoring for production readiness

## 📖 Documentation

- [**POST_RL_USAGE_GUIDE.md**](POST_RL_USAGE_GUIDE.md) - Advanced payload validation guide
- [**REFACTORING_SUMMARY.md**](REFACTORING_SUMMARY.md) - Code architecture documentation
- [**KAGGLE_SETUP.md**](KAGGLE_SETUP.md) - Cloud training setup guide

## 🔬 Research Applications

This framework is designed for:
- **Security Research**: Understanding WAF bypass techniques
- **Red Team Operations**: Generating evasive payloads for penetration testing
- **WAF Development**: Testing detection capabilities
- **ML Security**: Studying adversarial examples in NLP
- **Education**: Teaching web security and AI/ML integration

## ⚠️ Ethical Use & Disclaimer

**IMPORTANT**: This tool is intended for **authorized security testing and research purposes only**. Users must:

- Obtain proper authorization before testing any system
- Comply with applicable laws and regulations
- Use responsibly and ethically
- Not engage in malicious activities

The authors are not responsible for misuse or damage caused by this software.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook AI Research for OPT models
- Google for FLAN-T5 models
- Hugging Face for the Transformers library
- Open-source security research community

## 📧 Contact

For questions, suggestions, or collaboration:
- GitHub: [@JayBao2604](https://github.com/JayBao2604)
- Repository: [VALKYRIE-WAF](https://github.com/JayBao2604/VALKYRIE-WAF)

---

**Built with ❤️ for the security research community**

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
