import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(TwinCritic, self).__init__()
        # Critic 1
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)
        # Critic 2
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_out(x1)
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q2_out(x2)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_noisy=True):
        """
        Actor network.
         - input_size: dimension de l'état (par ex. 8000 pour un grid sensor 5x40x40 aplati)
         - hidden_size: taille des couches cachées (ex: 256)
         - output_size: dimension de l'action continue (ici 3)
         - use_noisy: booléen pour activer/désactiver les NoisyLinear
        """
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
        state: tensor de forme (batch, input_size)
        Retourne des actions dans [-1,1] de forme (batch, output_size)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x