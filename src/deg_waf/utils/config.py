"""Configuration and constants for training."""


def get_config(attack_type='sqli'):
    """Get configuration for a specific attack type."""
    return {
        "attack_type": attack_type,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 1.0,
        "max_length": 64,
        "num_train_iters": 500,
        "gamma": 0.99,
        "lam": 0.95,  # GAE parameter
        "save_interval": 50,
        "log_interval": 10,
        "reward_model_path": f"models/rewards/opt125m-reward_{attack_type}",
        "policy_model_path": f"models/pretrained/facebook_opt125m_{attack_type}",
        "vocab_path": f"data/generated/{attack_type}_vocab.json",
    }


# Attack type prompts mapping
ATTACK_PROMPTS = {
    'sqli': [
        "Generate SQLi payload:",
        "Create SQLi attack:",
        "Write SQLi code:",
        "Build SQLi payload:",
        "SQLi injection:",
        "JavaScript SQLi:",
        "SQLi exploit:",
        "SQLi vector:",
        "SQLi script:",
        "SQLi vulnerability:"
    ],
    'xss': [
        "Generate XSS payload:",
        "Create XSS attack:",
        "Write XSS code:",
        "Build XSS payload:",
        "XSS injection:",
        "JavaScript XSS:",
        "XSS exploit:",
        "XSS vector:",
        "XSS script:",
        "XSS vulnerability:"
    ],
    'rce': [
        "Generate RCE payload:",
        "Create RCE attack:",
        "Write RCE code:",
        "Build RCE payload:",
        "RCE injection:",
        "Command injection:",
        "RCE exploit:",
        "RCE vector:",
        "Shell command:",
        "RCE vulnerability:"
    ],
    'nosqli': [
        "Generate NoSQL injection:",
        "Create NoSQLi attack:",
        "Write NoSQLi code:",
        "Build NoSQLi payload:",
        "NoSQL injection:",
        "MongoDB injection:",
        "NoSQLi exploit:",
        "NoSQLi vector:",
        "NoSQLi script:",
        "NoSQL vulnerability:"
    ],
    'ssrf': [
        "Generate SSRF payload:",
        "Create SSRF attack:",
        "Write SSRF code:",
        "Build SSRF payload:",
        "SSRF injection:",
        "Server request:",
        "SSRF exploit:",
        "SSRF vector:",
        "SSRF URL:",
        "SSRF vulnerability:"
    ]
}
