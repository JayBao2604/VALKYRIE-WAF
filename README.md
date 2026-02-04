
# VALKYRIE-WAF 🛡️

> **Assessing the Robustness of Web Application Firewalls using Reinforcement Learning and Pre-trained Language Modelss**

## 🎥 Demo Video

https://github.com/user-attachments/assets/d82b5732-8185-433e-96af-743196dbf2f3



*Complete demonstration of VALKYRIE-WAF in action - showing payload generation, WAF bypass testing, and validation pipeline*

> **Note:** To properly display the video, please drag and drop the `[Demo]-VALKYRIE-WAF.mp4` file into a GitHub issue or this README edit box, then copy the generated URL here.

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
- Fine-tuned **Facebook OPT-125M** models for attack payload generation
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

## 🧪 Training Pipeline

### Stage 1: Pretraining
1. Load base language model (OPT-125M)
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




