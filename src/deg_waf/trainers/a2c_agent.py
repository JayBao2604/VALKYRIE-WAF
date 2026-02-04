"""A2C Agent for generating sequences and managing trajectory data."""

import torch


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
