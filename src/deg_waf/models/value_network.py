"""Value network for the critic in A2C algorithm."""

import torch
import torch.nn as nn


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
