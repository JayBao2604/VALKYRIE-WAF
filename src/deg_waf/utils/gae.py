"""Generalized Advantage Estimation (GAE) utilities."""

import torch


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
