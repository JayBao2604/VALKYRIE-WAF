"""Training modules for reinforcement learning."""

from .a2c_agent import A2CAgent
from .a2c_trainer import train_a2c

__all__ = ['A2CAgent', 'train_a2c']
