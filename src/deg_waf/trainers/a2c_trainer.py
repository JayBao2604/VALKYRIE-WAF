"""A2C training loop with GAE for attack payload generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import os
import pandas as pd
import json
import traceback

from deg_waf.models import load_reward_model, get_reward_score, ValueHead
from deg_waf.trainers import A2CAgent
from deg_waf.utils import compute_gae, normalize_advantages, ATTACK_PROMPTS


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
                save_path = f"models/finetuned/opt-a2c-{config['attack_type']}-best"
                os.makedirs(save_path, exist_ok=True)
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ New best model saved with reward: {best_reward:.4f}")

            # Periodic saving
            if iteration % config["save_interval"] == 0 and iteration > 0:
                save_path = f"models/finetuned/opt-a2c-{config['attack_type']}-iter{iteration}"
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
    final_save_path = f"models/finetuned/opt-a2c-{config['attack_type']}-final"
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
