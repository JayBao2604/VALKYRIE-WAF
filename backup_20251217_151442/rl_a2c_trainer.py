"""
A2C Reinforcement Learning Training Script with GAE
for Attack Payload Generation

This script implements an Advantage Actor-Critic (A2C) algorithm with
Generalized Advantage Estimation (GAE) to train a language model for
generating various attack payloads (SQLi, XSS, RCE, NoSQLi, SSRF) using a reward model.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig, OPTModel
from tqdm import tqdm
import random
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import traceback
import argparse
from pathlib import Path


# ============================ CONFIG ============================
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
        "reward_model_path": f"models/reward-models/modsecurity/sampler/opt125m-reward_{attack_type}",
        "policy_model_path": f"models/pretrain-models/facebook_opt125m_{attack_type}",
        "vocab_path": f"data/{attack_type}/{attack_type}_vocab.json",
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


# ============================ REWARD MODEL ============================
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


# ============================ CRITIC MODEL ============================
class ValueHead(nn.Module):
    """Value network for the critic in A2C algorithm."""

    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, hidden_states):
        if len(hidden_states.shape) == 3:
            last_hidden = hidden_states[:, -1, :]
        else:
            last_hidden = hidden_states
        return self.value(last_hidden).squeeze(-1)


# ============================ A2C AGENT WITH GAE ============================
class A2CAgent:
    """A2C Agent for generating sequences and managing trajectory data."""

    def __init__(self, policy_model, value_model, tokenizer, device, max_length=64):
        self.policy_model = policy_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def generate_with_trajectory(self, prompts):
        """Generate sequences with full trajectory for GAE."""
        batch_size = len(prompts)

        # Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batch_log_probs = []
        batch_values = []
        batch_entropies = []
        batch_hidden_states = []  # Store hidden states for value updates

        # Generate token-by-token
        for step in range(self.max_length - input_ids.shape[1]):
            # Forward pass policy with output_hidden_states
            outputs = self.policy_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # Calculate probability distributions
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Get log probabilities and entropy for chosen tokens
            token_log_probs = log_probs.gather(1, next_tokens)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

            # Get value estimates - use value model
            # DETACH hidden states to avoid backprop through policy model in value loss
            values = self.value_model(last_hidden.detach())

            # Store trajectory data
            batch_log_probs.append(token_log_probs)
            batch_values.append(values.unsqueeze(1))
            batch_entropies.append(entropy)
            batch_hidden_states.append(last_hidden.detach().unsqueeze(1))  # Detach here

            # Append new tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)

            # Check for EOS tokens
            eos_mask = (next_tokens == self.tokenizer.eos_token_id).squeeze()
            if eos_mask.any():
                if eos_mask.all():
                    break

        # Decode generated sequences
        generated_texts = []
        for i in range(batch_size):
            seq_input_ids = input_ids[i:i+1]
            generated_text = self.tokenizer.decode(seq_input_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Stack all trajectory data
        trajectory = {
            'generated_texts': generated_texts,
            'log_probs': torch.cat(batch_log_probs, dim=1) if batch_log_probs else torch.tensor([]).to(self.device),
            'values': torch.cat(batch_values, dim=1) if batch_values else torch.tensor([]).to(self.device),
            'entropies': torch.cat(batch_entropies, dim=1) if batch_entropies else torch.tensor([]).to(self.device),
            'hidden_states': torch.cat(batch_hidden_states, dim=1) if batch_hidden_states else torch.tensor([]).to(self.device),
            'sequence_lengths': torch.tensor([len(batch_log_probs)] * batch_size).to(self.device) if batch_log_probs else torch.tensor([]).to(self.device),
            'final_states': input_ids
        }

        return trajectory


# ============================ GAE CALCULATION ============================
def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards for each timestep
        values: List of value estimates for each timestep
        next_value: Value estimate for the next state after the final timestep
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: List of advantage estimates
        returns: List of return estimates
    """
    advantages = []
    gae = 0
    next_value = next_value

    # Calculate advantages backwards
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_value - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]

        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]

    return advantages, returns


def normalize_advantages(advantages):
    """Normalize advantages across the batch."""
    if isinstance(advantages, list):
        all_advantages = torch.tensor(advantages, dtype=torch.float32)
    else:
        all_advantages = advantages

    if all_advantages.std() > 0:
        normalized_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
    else:
        normalized_advantages = all_advantages - all_advantages.mean()

    return normalized_advantages


# ============================ MAIN TRAINING FUNCTION ============================
def train_a2c(config):
    """Main training loop for A2C with GAE."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    attack_type = config["attack_type"].upper()
    print(f"Using device: {device}")
    print(f"Training for attack type: {attack_type}")

    # Load models
    policy_model_path = config["policy_model_path"]
    reward_model_path = config["reward_model_path"]

    # Load tokenizer and policy model
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path).to(device)

    # Load special tokens from vocab
    try:
        vocab_path = config["vocab_path"]
        with open(vocab_path) as file:
            vocab = json.load(file)
        special_tokens = list(vocab.keys())
        tokenizer.add_tokens(special_tokens)
        policy_model.resize_token_embeddings(len(tokenizer))
        print(f"Added {len(special_tokens)} special tokens from {vocab_path}")
    except Exception as e:
        print(f"No special tokens found at {config.get('vocab_path', 'N/A')}, using base tokenizer: {e}")

    tokenizer.pad_token = tokenizer.eos_token
    policy_model.config.pad_token_id = tokenizer.eos_token_id

    # Load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = load_reward_model(reward_model_path, device)
    print("✅ Reward model loaded successfully")

    # Initialize value model
    hidden_size = policy_model.config.hidden_size
    value_model = ValueHead(hidden_size).to(device)

    # Create A2C agent
    agent = A2CAgent(policy_model, value_model, tokenizer, device, config["max_length"])

    # Optimizers
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config["learning_rate"])
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=config["learning_rate"])

    # Get prompts for this attack type
    attack_prompts = ATTACK_PROMPTS.get(config["attack_type"], ATTACK_PROMPTS['sqli'])

    # Training statistics
    best_reward = 0.0
    reward_history = []

    print(f"🚀 Starting A2C Training with GAE for {attack_type}...")

    for iteration in tqdm(range(config["num_train_iters"]), desc=f"A2C {attack_type} Training"):
        try:
            # Sample batch prompts
            batch_prompts = random.choices(attack_prompts, k=config["batch_size"])

            # Generate sequences with trajectory
            trajectory = agent.generate_with_trajectory(batch_prompts)

            if trajectory['log_probs'].numel() == 0:
                continue

            # Calculate final rewards using the trained reward model
            final_rewards = []
            for text in trajectory['generated_texts']:
                reward = get_reward_score(reward_model, reward_tokenizer, text, device)
                final_rewards.append(reward)

            final_rewards = torch.tensor(final_rewards, device=device)

            # Get sequence lengths
            seq_lengths = trajectory['sequence_lengths']

            # Prepare data for GAE
            batch_advantages = []
            batch_returns = []
            all_log_probs = []
            all_entropies = []
            all_hidden_states = []

            # Calculate next values for final states
            with torch.no_grad():
                final_hidden = policy_model(
                    trajectory['final_states'],
                    attention_mask=torch.ones_like(trajectory['final_states']),
                    output_hidden_states=True
                ).hidden_states[-1][:, -1, :]
                next_values = value_model(final_hidden)

            # Process each sequence in the batch
            for i in range(config["batch_size"]):
                seq_len = seq_lengths[i].item()
                if seq_len == 0:
                    continue

                # Create sparse rewards (only final reward)
                rewards = [0.0] * (seq_len - 1) + [final_rewards[i].item()]

                # Get values for this sequence
                values = trajectory['values'][i, :seq_len].cpu().detach().numpy()

                # Calculate GAE
                advantages, returns = compute_gae(
                    rewards,
                    values,
                    next_values[i].item(),
                    gamma=config["gamma"],
                    lam=config["lam"]
                )

                batch_advantages.extend(advantages)
                batch_returns.extend(returns)

                # Collect corresponding log_probs, entropies and hidden states
                all_log_probs.append(trajectory['log_probs'][i, :seq_len])
                all_entropies.append(trajectory['entropies'][i, :seq_len])
                all_hidden_states.append(trajectory['hidden_states'][i, :seq_len])

            if not batch_advantages:
                continue

            # Convert to tensors
            advantages_tensor = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
            returns_tensor = torch.tensor(batch_returns, device=device, dtype=torch.float32)
            log_probs_tensor = torch.cat(all_log_probs)
            entropies_tensor = torch.cat(all_entropies)
            hidden_states_tensor = torch.cat(all_hidden_states)

            # Normalize advantages
            advantages_tensor = normalize_advantages(advantages_tensor)

            # ============================ POLICY UPDATE ============================
            # Policy loss
            policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
            entropy_loss = -entropies_tensor.mean()
            total_policy_loss = policy_loss + config["entropy_coef"] * entropy_loss

            # Update policy model
            policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config["max_grad_norm"])
            policy_optimizer.step()

            # ============================ VALUE UPDATE ============================
            # Value loss - use hidden states that were DETACHED
            current_value_predictions = value_model(hidden_states_tensor)

            # Value loss: value predictions should match returns
            value_loss = ((current_value_predictions - returns_tensor) ** 2).mean()
            total_value_loss = config["value_loss_coef"] * value_loss

            # Update value model
            value_optimizer.zero_grad()
            total_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), config["max_grad_norm"])
            value_optimizer.step()

            # Logging
            avg_reward = final_rewards.mean().item()
            reward_history.append(avg_reward)

            if iteration % config["log_interval"] == 0:
                print(f"\nIteration {iteration}:")
                print(f"Avg Reward: {avg_reward:.4f}")
                print(f"Policy Loss: {policy_loss.item():.4f}")
                print(f"Value Loss: {value_loss.item():.4f}")
                print(f"Entropy: {entropy_loss.item():.4f}")
                print(f"GAE Advantage Mean: {advantages_tensor.mean().item():.4f}")
                print(f"GAE Advantage Std: {advantages_tensor.std().item():.4f}")

                # Show sample generation
                sample_idx = random.randint(0, len(batch_prompts)-1)
                sample_text = trajectory['generated_texts'][sample_idx]
                sample_reward = final_rewards[sample_idx].item()
                print(f"Sample: {batch_prompts[sample_idx]}")
                print(f"Generated: {sample_text[:80]}...")
                print(f"Reward: {sample_reward:.4f}")
                print("-" * 50)

            # Save best model
            if avg_reward > best_reward and iteration > 10:
                best_reward = avg_reward
                save_path = f"models/finetune-models/opt-a2c-{config['attack_type']}-best-gae"
                os.makedirs(save_path, exist_ok=True)
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ New best model saved with reward: {best_reward:.4f}")

            # Periodic saving
            if iteration % config["save_interval"] == 0 and iteration > 0:
                save_path = f"models/finetune-models/opt-a2c-{config['attack_type']}-iter{iteration}-gae"
                os.makedirs(save_path, exist_ok=True)
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # Save training history
                history_df = pd.DataFrame({
                    'iteration': range(len(reward_history)),
                    'reward': reward_history
                })
                history_df.to_csv(f"{save_path}/training_history.csv", index=False)
                print(f"✅ Model saved at iteration {iteration}")

        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            traceback.print_exc()
            continue

    # Final save
    final_save_path = f"models/finetune-models/opt-a2c-{config['attack_type']}-final-gae"
    os.makedirs(final_save_path, exist_ok=True)
    policy_model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # Save final training history
    history_df = pd.DataFrame({
        'iteration': range(len(reward_history)),
        'reward': reward_history
    })
    history_df.to_csv(f"{final_save_path}/training_history.csv", index=False)

    print(f"✅ Final A2C model with GAE saved to: {final_save_path}")
    print(f"📊 Best reward achieved: {best_reward:.4f}")
    print(f"📈 Training completed with {len(reward_history)} iterations")


# ============================ ENTRY POINT ============================
def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Train A2C model for attack payload generation')
    parser.add_argument('-t', '--attack-type', type=str, 
                        choices=['sqli', 'xss', 'rce', 'nosqli', 'ssrf'],
                        default='sqli',
                        help='Type of attack to train for (default: sqli)')
    parser.add_argument('-n', '--num-iters', type=int, default=500,
                        help='Number of training iterations (default: 500)')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--max-length', type=int, default=64,
                        help='Maximum sequence length (default: 64)')
    parser.add_argument('--policy-model', type=str,
                        help='Path to policy model (overrides default)')
    parser.add_argument('--reward-model', type=str,
                        help='Path to reward model (overrides default)')
    parser.add_argument('--vocab-path', type=str,
                        help='Path to vocabulary file (overrides default)')
    
    args = parser.parse_args()
    
    # Get config for attack type
    config = get_config(args.attack_type)
    
    # Override config with CLI arguments
    config['num_train_iters'] = args.num_iters
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['max_length'] = args.max_length
    
    if args.policy_model:
        config['policy_model_path'] = args.policy_model
    if args.reward_model:
        config['reward_model_path'] = args.reward_model
    if args.vocab_path:
        config['vocab_path'] = args.vocab_path
    
    print(f"{'='*60}")
    print(f"A2C Training Configuration for {args.attack_type.upper()}")
    print(f"{'='*60}")
    print(f"Attack Type: {config['attack_type']}")
    print(f"Iterations: {config['num_train_iters']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Max Length: {config['max_length']}")
    print(f"Policy Model: {config['policy_model_path']}")
    print(f"Reward Model: {config['reward_model_path']}")
    print(f"Vocab Path: {config['vocab_path']}")
    print(f"{'='*60}\n")
    
    # Start training
    train_a2c(config)


if __name__ == "__main__":
    main()
