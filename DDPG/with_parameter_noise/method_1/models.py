import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        state, action sont des tenseurs torch (batch_size, state_dim), (batch_size, action_dim).
        On concatène => (batch_size, state_dim + action_dim).
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_noisy=True):
        super(Actor, self).__init__()
        if use_noisy:
            self.linear1 = NoisyLinear(input_size, hidden_size)
            self.linear2 = NoisyLinear(hidden_size, hidden_size)
            self.linear3 = NoisyLinear(hidden_size, output_size)
        else:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        state: Tensor de forme (batch_size, input_size)
        Retourne des actions dans [-1, 1]
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # Utilisation de tanh pour borner l'action dans [-1, 1]
        x = torch.tanh(self.linear3(x))
        return x
